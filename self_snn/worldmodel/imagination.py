from dataclasses import dataclass
from typing import Dict, Any

import torch


@dataclass
class ImaginationConfig:
    horizon: int = 10


class ImaginationEngine:
    def __init__(self, config: ImaginationConfig) -> None:
        self.config = config

    def rollout(self, z: torch.Tensor, h: torch.Tensor) -> Dict[str, Any]:
        traj = [h]
        cur = h
        for _ in range(self.config.horizon):
            cur = cur + 0.1 * torch.tanh(z)
            traj.append(cur)
        return {"traj": torch.stack(traj, dim=0)}

