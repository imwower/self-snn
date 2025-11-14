from dataclasses import dataclass
from typing import Dict, Any

import torch


@dataclass
class EpisodicIndexConfig:
    max_episodes: int = 1024


class EpisodicIndex:
    def __init__(self, config: EpisodicIndexConfig) -> None:
        self.config = config
        self._index: Dict[int, Dict[str, Any]] = {}
        self._next_id: int = 0

    def add(self, key: torch.Tensor, summary: str) -> int:
        eid = self._next_id
        self._next_id += 1
        if len(self._index) >= self.config.max_episodes:
            oldest = min(self._index.keys())
            self._index.pop(oldest, None)
        self._index[eid] = {"key": key.detach().cpu(), "summary": summary}
        return eid

    def get(self, eid: int) -> Dict[str, Any] | None:
        return self._index.get(eid)

