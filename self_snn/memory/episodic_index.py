from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch


@dataclass
class EpisodicIndexConfig:
    """
    事件索引配置。

    Parameters
    ----------
    max_episodes:
        最大存储的事件数量（超过后按 LRU 删除）。
    ttl_steps:
        时间步 TTL（写入步数差超过该值的事件将被清理）。默认为 None 表示不启用 TTL。
    """

    max_episodes: int = 1024
    ttl_steps: Optional[int] = None


class EpisodicIndex:
    """
    事件级索引：以 Self-Key / Context-Key / Phase 组合键为索引，可做 LRU/TTL 管理与命中率统计。
    """

    def __init__(self, config: EpisodicIndexConfig) -> None:
        self.config = config
        self._index: Dict[int, Dict[str, Any]] = {}
        self._next_id: int = 0
        self._step: int = 0
        self._lookups: int = 0
        self._hits: int = 0

    @staticmethod
    def build_key(self_key: torch.Tensor, context_key: Optional[torch.Tensor] = None, phase: float = 0.0) -> torch.Tensor:
        """
        构造 EpisodicIndex 的复合键：concat(Self-Key, Context-Key, Phase)。
        """
        parts = [self_key.flatten()]
        if context_key is not None:
            parts.append(context_key.flatten())
        phase_tensor = torch.tensor([phase], dtype=self_key.dtype, device=self_key.device)
        parts.append(phase_tensor)
        return torch.cat(parts, dim=0)

    def add(self, key: torch.Tensor, summary: str) -> int:
        """
        新增一个事件条目，返回 episode id。
        """
        self._step += 1
        eid = self._next_id
        self._next_id += 1

        # TTL 清理：删除过旧的事件
        if self.config.ttl_steps is not None:
            cutoff = self._step - max(self.config.ttl_steps, 0)
            self._index = {
                i: info for i, info in self._index.items() if info.get("created_step", 0) >= cutoff
            }

        # LRU：超过上限时移除最早写入的条目
        if len(self._index) >= self.config.max_episodes:
            oldest_id = min(self._index.keys(), key=lambda i: self._index[i].get("created_step", i))
            self._index.pop(oldest_id, None)

        self._index[eid] = {
            "key": key.detach().cpu(),
            "summary": summary,
            "created_step": self._step,
            "hits": 0,
        }
        return eid

    def get(self, eid: int) -> Optional[Dict[str, Any]]:
        """
        按 episode id 检索事件，更新命中率统计。
        """
        self._lookups += 1
        info = self._index.get(eid)
        if info is None:
            return None

        info["hits"] = info.get("hits", 0) + 1
        self._hits += 1
        return info

    def hit_rate(self) -> float:
        """
        返回当前索引命中率（hits / lookups）。
        """
        if self._lookups <= 0:
            return 0.0
        return float(self._hits) / float(self._lookups)
