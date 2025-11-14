from dataclasses import dataclass
from typing import Tuple, List

import torch

from ..memory.delay_mem import DelayMemory
from ..worldmodel.pred_code import PredictiveCoder
from .self_model import SelfModel


@dataclass
class IntentionConfig:
    n_candidates: int = 5
    horizon: int = 10
    w_lp: float = 0.45
    w_empower: float = 0.2
    w_reward: float = 0.15
    c_energy: float = 0.1
    c_risk: float = 0.07
    c_boredom: float = 0.03


class IntentionModule:
    def __init__(self, config: IntentionConfig) -> None:
        self.config = config

    def _utility(self, g: torch.Tensor) -> torch.Tensor:
        lp = g[0]
        empower = g[1].abs()
        reward = g[2]
        energy = g[3].abs()
        risk = g[4].abs()
        boredom = g[5].abs()
        return (
            self.config.w_lp * lp
            + self.config.w_empower * empower
            + self.config.w_reward * reward
            - self.config.c_energy * energy
            - self.config.c_risk * risk
            - self.config.c_boredom * boredom
        )

    def __call__(
        self, memory: DelayMemory, pred: PredictiveCoder, self_model: SelfModel
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        goals: List[torch.Tensor] = []
        utilities: List[torch.Tensor] = []
        base = self_model.key[:6].float()
        for _ in range(self.config.n_candidates):
            noise = 0.1 * torch.randn_like(base)
            g = base + noise
            u = self._utility(g)
            goals.append(g)
            utilities.append(u)
        return goals, torch.stack(utilities, dim=0)

