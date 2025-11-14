from __future__ import annotations

from dataclasses import dataclass, field
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
    def __init__(self, config: DelayMemoryConfig) -> None:
        self.config = config
        self._storage: Dict[Tuple[int, ...], List[torch.Tensor]] = {}
        self._delays: Dict[Tuple[int, ...], int] = {}

    @staticmethod
    def _key_to_tuple(key: torch.Tensor) -> Tuple[int, ...]:
        return tuple((key.detach().cpu().flatten() > 0).int().tolist())

    def write(self, key: torch.Tensor, sequence: torch.Tensor, delay: int | None = None) -> None:
        tkey = self._key_to_tuple(key)
        self._storage.setdefault(tkey, []).append(sequence.detach().clone())
        if delay is not None:
            self._delays[tkey] = int(max(0, min(self.config.dmax, delay)))

    def read(self, key: torch.Tensor) -> torch.Tensor | None:
        tkey = self._key_to_tuple(key)
        seqs = self._storage.get(tkey)
        if not seqs:
            return None
        stacked = torch.stack(seqs, dim=0)
        return stacked.mean(dim=0)

    def replay(self, key: torch.Tensor) -> torch.Tensor | None:
        seq = self.read(key)
        if seq is None:
            return None
        delay = self._delays.get(self._key_to_tuple(key), 0)
        if delay <= 0:
            return seq
        pad = torch.zeros(delay, *seq.shape[1:], dtype=seq.dtype, device=seq.device)
        return torch.cat([pad, seq], dim=0)[: self.config.dmax]

    def consolidate(self) -> None:
        # 简单“睡眠巩固”：同一键下的多次写入取平均，只保留 1 条
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
