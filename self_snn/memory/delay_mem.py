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
        读取该键下所有序列的平均值，作为“巩固后”表征。
        """
        tkey = self._key_to_tuple(key)
        seqs = self._storage.get(tkey)
        if not seqs:
            return None
        stacked = torch.stack(seqs, dim=0)
        return stacked.mean(dim=0)

    def replay(self, key: torch.Tensor) -> torch.Tensor | None:
        """
        按当前估计的延迟 d 进行重放：在时间前部填充 d 步空白。
        """
        seq = self.read(key)
        if seq is None:
            return None
        delay = self._delays.get(self._key_to_tuple(key), 0)
        if delay <= 0:
            return seq
        pad = torch.zeros(delay, *seq.shape[1:], dtype=seq.dtype, device=seq.device)
        return torch.cat([pad, seq], dim=0)[: self.config.dmax]

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
        cur_d = self._delays.get(tkey, 0)

        target_d = self._estimate_best_delay(spike_counts)

        # HDP 更新步长：朝着 target_d 以 delta_step 为单位移动，受第三因子缩放
        delta = target_d - cur_d
        if delta == 0:
            self._delays[tkey] = int(max(0, min(self.config.dmax, cur_d)))
            return

        direction = 1 if delta > 0 else -1
        step = direction * self.config.delta_step * float(third_factor)

        # 将第三因子缩放后取整，至少走一步
        if step == 0:
            step = direction * self.config.delta_step

        new_d = int(cur_d + step)
        new_d = max(0, min(self.config.dmax, new_d))
        self._delays[tkey] = new_d
