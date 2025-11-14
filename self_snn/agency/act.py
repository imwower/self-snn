from dataclasses import dataclass
from typing import Dict, Any

import torch


@dataclass
class ActConfig:
    router_topk: int = 2
    mpc_horizon: int = 5
    efference_gain: float = 0.7


class ActModule:
    def __init__(self, config: ActConfig) -> None:
        self.config = config

    def __call__(self, commit_state: Dict[str, Any], wm_state: torch.Tensor, gw_mask: torch.Tensor) -> Dict[str, Any]:
        committed = commit_state.get("committed", False)
        action = None
        if committed and commit_state.get("goal") is not None:
            goal = commit_state["goal"]
            action = torch.tanh(goal.mean() * wm_state.mean())
        else:
            action = torch.tensor(0.0, device=wm_state.device)
        efference = self.config.efference_gain * action
        energy = float(wm_state.abs().sum())
        return {"action": action, "efference": efference, "energy": energy}

