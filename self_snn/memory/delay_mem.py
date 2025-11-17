from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class DelayMemoryConfig:
    dmax: int = 120
    dt_ms: float = 1.0
    eta_d: float = 0.1
    delta_step: int = 1
    delay_entropy_reg: float = 1e-4


class DelayMemory:
    """
    延迟记忆模块（D-MEM）：
    - 对每个键存储若干时序片段 sequence（通常为 WM 或 feature 轨迹）
    - 为该键维护一个可塑的整数延迟 d \in [0, dmax]
    - 提供 write/read/replay/consolidate API
    - 提供基于 spike 自相关的简化 HDP 更新：update_delay_from_spikes
    """

    def __init__(self, config: DelayMemoryConfig) -> None:
        self.config = config
        self._storage: Dict[Tuple[int, ...], List[torch.Tensor]] = {}
        self._delays: Dict[Tuple[int, ...], int] = {}

    @staticmethod
    def _key_to_tuple(key: torch.Tensor) -> Tuple[int, ...]:
        return tuple((key.detach().cpu().flatten() > 0).int().tolist())

    def write(self, key: torch.Tensor, sequence: torch.Tensor, delay: int | None = None) -> None:
        """
        追加写入一个时序序列到给定键。
        可选地显式指定 delay（例如在有监督设置下已知真实延迟）。
        """
        tkey = self._key_to_tuple(key)
        self._storage.setdefault(tkey, []).append(sequence.detach().clone())
        if delay is not None:
            self._delays[tkey] = int(max(0, min(self.config.dmax, delay)))

    def read(self, key: torch.Tensor) -> torch.Tensor | None:
        """
        读取该键下所有序列在时间与样本维度上的平均值，
        作为“巩固后”的紧凑表征向量。
        """
        tkey = self._key_to_tuple(key)
        seqs = self._storage.get(tkey)
        if not seqs:
            return None
        stacked = torch.stack(seqs, dim=0)  # [n_seq, T, ...]
        # 对 n_seq 与 T 两个维度求均值，返回形状为特征维度的向量
        return stacked.mean(dim=(0, 1))

    def replay(self, key: torch.Tensor) -> torch.Tensor | None:
        """
        按当前估计的延迟 d 进行重放。

        为了贴近“事件级环形缓冲 + scatter_add_”语义，这里在调用时构造一个长度 <= dmax
        的缓冲区 buffer，并将原始序列按 (t + d) 的时间索引散射累加：

            buffer[t + d] += seq[t]

        这样既保持了 DelayMemory 在本项目中的简化接口，又能在需要时反映“离散延迟链”的效果。
        """
        tkey = self._key_to_tuple(key)
        seqs = self._storage.get(tkey)
        if not seqs:
            return None
        # 对同一键的多条写入在样本维度求平均，保留时间维度
        seq = torch.stack(seqs, dim=0).mean(dim=0)  # [T, ...]

        delay = int(self._delays.get(tkey, 0))
        dmax = int(self.config.dmax)

        # 原始序列长度
        T = int(seq.shape[0])
        if T <= 0:
            return None

        # 输出长度：最多到 dmax，避免越界
        out_len = min(T + max(delay, 0), dmax)
        device = seq.device
        buf = torch.zeros(out_len, *seq.shape[1:], dtype=seq.dtype, device=device)

        # 目标时间索引 t + d，并裁剪到 [0, out_len)
        t_idx = torch.arange(T, device=device)
        dst = t_idx + max(delay, 0)
        valid = dst < out_len
        if not torch.any(valid):
            return buf
        dst = dst[valid]
        src = seq[t_idx[valid]]

        # scatter_add_ / index_add_ 形式的事件级写入
        buf.index_add_(0, dst, src)
        return buf

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

        - spike_counts: shape [T]，为某一键相关联的 pre spike 总和轨迹
        - third_factor: 第三因子，范围大致 [-1, 1]，决定更新方向与步长
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
