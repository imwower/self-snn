from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EPropHeadConfig:
    input_dim: int = 128
    output_dim: int = 1


class EPropHead(nn.Module):
    def __init__(self, config: EPropHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.out = nn.Linear(config.input_dim, config.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.out(x)

