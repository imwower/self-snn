from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Tuple as Tup

import torch

from ..meta.plasticity import STDPConfig, STDPState, stdp_3factor


@dataclass
class DelayMemoryConfig:
    """
    延迟记忆（D-MEM）配置。

    Parameters
    ----------
    dmax:
        最大延迟窗口长度（以时间步计），同时也是环形缓冲长度上界。
    dt_ms:
        时间步长（ms），用于将步数转换为物理时间。
    eta_d:
        延迟可塑性学习率（HDP 中第三因子缩放系数）。
    delta_step:
        单步延迟更新的最大步长，保证 Δd ∈ {-delta_step, 0, +delta_step}。
    delay_entropy_reg:
        延迟分布的熵正则系数（当前实现中作为占位，便于后续引入）。
    """

    dmax: int = 120
    dt_ms: float = 1.0
    eta_d: float = 0.1
    delta_step: int = 1
    delay_entropy_reg: float = 1e-4


class DelayMemory:
    """
    延迟记忆模块（D-MEM）。

    - 对每个键存储若干时序片段（通常为工作记忆或世界模型隐状态轨迹）。
    - 为该键维护一个可塑的整数延迟 d ∈ [0, dmax]。
    - 使用事件级环形缓冲与 `scatter_add_` 实现重放。
    - 提供基于 pre spike 自相关与第三因子（RPE/Surprise/Empowerment 等）的
      简化 HDP 更新：`update_delay_from_spikes`。

    Notes
    -----
    - 为兼容 tests 与现有脚本：
      - `read(key)` 返回紧凑表征向量（时间与写入次数双均值）。
      - `replay(key)` 返回按当前 d 重放后的时序张量。
    - 额外提供 `read_with_timing_error(key, max_len)` 接口，用于评估重放时间误差。
    """

    def __init__(self, config: DelayMemoryConfig) -> None:
        self.config = config
        self._storage: Dict[Tuple[int, ...], List[torch.Tensor]] = {}
        self._delays: Dict[Tuple[int, ...], int] = {}
        # STDP 权重与资格迹状态（按键存储）
        self._weights: Dict[Tuple[int, ...], torch.Tensor] = {}
        self._stdp_states: Dict[Tuple[int, ...], STDPState] = {}

    @staticmethod
    def _key_to_tuple(key: torch.Tensor) -> Tuple[int, ...]:
        return tuple((key.detach().cpu().flatten() > 0).int().tolist())

    def write(self, key: torch.Tensor, sequence: torch.Tensor, delay: Optional[int] = None) -> None:
        """
        追加写入一个时序序列到给定键。

        Parameters
        ----------
        key:
            键张量（将被二值化为 tuple，作为索引）。
        sequence:
            形状为 [T, ...] 的时序张量。
        delay:
            可选的显式延迟标签（例如有监督设定中已知真实延迟）。
        """
        tkey = self._key_to_tuple(key)
        self._storage.setdefault(tkey, []).append(sequence.detach().clone())
        if delay is not None:
            self._delays[tkey] = int(max(0, min(self.config.dmax, delay)))

    def read(self, key: torch.Tensor) -> Optional[torch.Tensor]:
        """
        读取该键下所有序列在时间与样本维度上的平均值，
        作为“巩固后”的紧凑表征向量。

        兼容 tests/test_delay_mem.py 对 `read(key)` 的使用。
        """
        tkey = self._key_to_tuple(key)
        seqs = self._storage.get(tkey)
        if not seqs:
            return None
        stacked = torch.stack(seqs, dim=0)  # [n_seq, T, ...]
        # 对 n_seq 与 T 两个维度求均值，返回形状为特征维度的向量
        return stacked.mean(dim=(0, 1))

    def _build_replay_buffer(self, tkey: Tuple[int, ...]) -> Optional[torch.Tensor]:
        """
        内部工具：根据当前延迟构造环形缓冲重放结果。
        """
        seqs = self._storage.get(tkey)
        if not seqs:
            return None
        seq = torch.stack(seqs, dim=0).mean(dim=0)  # [T, ...]

        delay = int(self._delays.get(tkey, 0))
        dmax = int(self.config.dmax)

        T = int(seq.shape[0])
        if T <= 0:
            return None

        out_len = min(T + max(delay, 0), dmax)
        device = seq.device
        buf = torch.zeros(out_len, *seq.shape[1:], dtype=seq.dtype, device=device)

        t_idx = torch.arange(T, device=device)
        dst = t_idx + max(delay, 0)
        valid = dst < out_len
        if not torch.any(valid):
            return buf
        dst = dst[valid]
        src = seq[t_idx[valid]]

        buf.index_add_(0, dst, src)
        return buf

    def replay(self, key: torch.Tensor) -> Optional[torch.Tensor]:
        """
        按当前估计的延迟 d 进行重放，返回时序张量。

        兼容 tests/test_delay_mem.py 对 `replay(key)` 的使用。
        """
        tkey = self._key_to_tuple(key)
        return self._build_replay_buffer(tkey)

    def read_with_timing_error(
        self, key: torch.Tensor, max_len: Optional[int] = None
    ) -> Tup[Optional[torch.Tensor], float]:
        """
        读取并重放，同时估计“时间重放误差”。

        当前实现中，误差以“估计延迟与当前延迟的差值”近似：

            err_ms ≈ |d_est - d_cur| * dt_ms

        后续可替换为基于 pre/post spike 对齐的 RMS 误差。
        """
        tkey = self._key_to_tuple(key)
        replayed = self._build_replay_buffer(tkey)
        if replayed is None:
            return None, 0.0

        cur_d = int(self._delays.get(tkey, 0))
        # 用自相关重新估计延迟，作为“理想值”
        # 这里仅使用单通道总和作为 spike_counts 的近似
        spike_counts = replayed.float().sum(dim=1)
        d_est = self._estimate_best_delay(spike_counts)
        err_steps = abs(d_est - cur_d)
        err_ms = float(err_steps * self.config.dt_ms)

        if max_len is not None and replayed.shape[0] > max_len:
            replayed = replayed[:max_len]

        return replayed, err_ms

    def consolidate(self) -> None:
        """
        简单“睡眠巩固”：同一键下的多次写入取平均，只保留 1 条。
        """
        for k, seqs in list(self._storage.items()):
            if len(seqs) <= 1:
                continue
            stacked = torch.stack(seqs, dim=0)
            mean_seq = stacked.mean(dim=0)
            self._storage[k] = [mean_seq]

    def erase(self, key: torch.Tensor) -> None:
        tkey = self._key_to_tuple(key)
        self._storage.pop(tkey, None)
        self._delays.pop(tkey, None)
        self._weights.pop(tkey, None)
        self._stdp_states.pop(tkey, None)

    # ---- HDP 风格延迟可塑性（简化版） ----

    def _estimate_best_delay(self, spike_counts: torch.Tensor) -> int:
        """
        根据 pre spike 自相关估计“最合适”的延迟：
        找到 d 使得 pre(t) 与 pre(t+d) 的相关性最大。
        """
        T = int(spike_counts.numel())
        if T <= 1:
            return 0
        max_d = min(self.config.dmax, T - 1)
        best_d = 0
        best_score = torch.tensor(0.0, device=spike_counts.device)
        for d in range(1, max_d + 1):
            s = (spike_counts[:-d] * spike_counts[d:]).sum()
            if s > best_score:
                best_score = s
                best_d = d
        return int(best_d)

    def update_delay_from_spikes(
        self, key: torch.Tensor, spike_counts: torch.Tensor, third_factor: float = 1.0
    ) -> None:
        """
        简化 HDP：基于 pre spike 自相关与第三因子（如 RPE/置信度）更新离散延迟。

        Parameters
        ----------
        key:
            与特定记忆槽关联的键。
        spike_counts:
            形状为 [T] 的张量，为该键相关联的 pre spike 总和轨迹。
        third_factor:
            第三因子，范围大致 [-1, 1]，决定更新方向与步长。
        """
        if spike_counts.numel() <= 1:
            return

        tkey = self._key_to_tuple(key)
        cur_d = int(self._delays.get(tkey, 0))

        target_d = self._estimate_best_delay(spike_counts)

        # HDP 更新：朝着 target_d 以单步 ±delta_step 变化，满足 Δd ∈ {-1, 0, +1}·delta_step
        delta = int(target_d - cur_d)
        # 第三因子过小或目标已对齐时，不更新
        if delta == 0 or abs(third_factor) < 1e-3:
            self._delays[tkey] = int(max(0, min(self.config.dmax, cur_d)))
            return

        direction = 1 if delta > 0 else -1
        step = direction * int(self.config.delta_step)

        new_d = cur_d + step
        new_d = max(0, min(int(self.config.dmax), new_d))
        self._delays[tkey] = int(new_d)

    # ---- STDP × 第三因子权重更新（可选） ----

    def update_weights(
        self,
        key: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        stdp_config: STDPConfig,
        third_factor: float | torch.Tensor,
        dt: float = 1.0,
    ) -> None:
        """
        对给定键的“记忆权重”执行三因子 STDP 更新。

        Notes
        -----
        - 该权重矩阵可由上层（如 SelfSNN）在每个时间步调用，用于将延迟记忆与
          STDP 资格迹 / 第三因子真正耦合。
        - 是否在读出路径中显式使用这些权重由上层决定，这里仅负责维护可塑性状态。
        """
        tkey = self._key_to_tuple(key)
        pre = pre.detach().flatten()
        post = post.detach().flatten()

        if tkey not in self._weights:
            self._weights[tkey] = torch.zeros(
                pre.numel(),
                post.numel(),
                dtype=pre.dtype,
                device=pre.device,
            )
            self._stdp_states[tkey] = STDPState(
                pre_trace=torch.zeros_like(pre),
                post_trace=torch.zeros_like(post),
            )

        w = self._weights[tkey]
        state = self._stdp_states[tkey]
        new_w, new_state = stdp_3factor(
            w,
            pre_spikes=pre,
            post_spikes=post,
            state=state,
            config=stdp_config,
            third_factor=third_factor,
            dt=dt,
        )
        self._weights[tkey] = new_w
        self._stdp_states[tkey] = new_state

    def delay_of(self, key: torch.Tensor) -> int:
        """
        返回给定键当前估计的整数延迟 d。
        """
        tkey = self._key_to_tuple(key)
        return int(self._delays.get(tkey, 0))

    def stats(self) -> Dict[str, float]:
        """
        返回全局延迟统计信息，便于监控或可视化。
        """
        if not self._delays:
            return {
                "n_keys": 0.0,
                "mean_delay": 0.0,
                "var_delay": 0.0,
                "entropy": 0.0,
                "entropy_reg": 0.0,
            }

        delays = torch.tensor(list(self._delays.values()), dtype=torch.float32)
        mean_delay = float(delays.mean())
        var_delay = float(delays.var(unbiased=False))

        # 以离散直方图近似延迟分布熵，用于延迟分布的熵正则项
        dmax = int(self.config.dmax)
        bins = torch.clamp(delays.long(), min=0, max=dmax)
        hist = torch.bincount(bins, minlength=dmax + 1).float()
        p = hist / hist.sum()
        mask = p > 0
        entropy = float((-p[mask] * p[mask].log()).sum().item())
        entropy_reg = float(self.config.delay_entropy_reg * entropy)

        return {
            "n_keys": float(len(self._delays)),
            "mean_delay": mean_delay,
            "var_delay": var_delay,
            "entropy": entropy,
            "entropy_reg": entropy_reg,
        }
