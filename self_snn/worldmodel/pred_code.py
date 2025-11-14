from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class PredictiveCoderConfig:
    hidden_dim: int = 128


class PredictiveCoder(nn.Module):
    def __init__(self, config: PredictiveCoderConfig) -> None:
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def forward(self, wm_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if wm_state.dim() == 1:
            x = wm_state
        else:
            x = wm_state.view(-1)
        if x.numel() != self.config.hidden_dim:
            x = nn.functional.pad(x, (0, self.config.hidden_dim - x.numel()))
            x = x[: self.config.hidden_dim]
        pred = self.net(x)
        err = pred - x
        return pred, err

