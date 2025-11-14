from dataclasses import dataclass
from typing import Dict, Any, List

import torch


@dataclass
class CommitConfig:
    threshold: float = 0.15
    min_interval_s: float = 3.0
    energy_cap: float = 2.0e6
    risk_cap: float = 0.6


class CommitModule:
    def __init__(self, config: CommitConfig) -> None:
        self.config = config
        self._last_commit_step: int = -10**9
        self._step: int = 0

    def __call__(
        self, goals: List[torch.Tensor], utilities: torch.Tensor, meta: Dict[str, Any], self_model: Any
    ) -> Dict[str, Any]:
        self._step += 1
        if utilities.numel() == 0:
            return {"committed": False}
        best_val, best_idx = utilities.max(dim=0)
        energy = self_model.state.get("energy", 0.0)
        risk = self_model.state.get("risk", 0.0)
        can_commit = (
            best_val.item() > self.config.threshold
            and energy < self.config.energy_cap
            and risk < self.config.risk_cap
        )
        committed = bool(can_commit)
        goal = goals[int(best_idx)] if committed else None
        return {"committed": committed, "goal": goal, "utility": best_val.detach()}

