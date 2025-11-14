from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from ..utils.masks import MaskLinear


@dataclass
class ExpertsConfig:
    input_dim: int = 128
    output_dim: int = 128
    num_experts: int = 16


class MaskedExperts(nn.Module):
    def __init__(self, config: ExpertsConfig) -> None:
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [MaskLinear(config.input_dim, config.output_dim, bias=True) for _ in range(config.num_experts)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outs = []
        synops = []
        for idx, expert in enumerate(self.experts):
            if mask[idx] <= 0:
                outs.append(torch.zeros_like(x))
                synops.append(torch.tensor(0.0, device=x.device))
            else:
                y = expert(x)
                outs.append(y)
                synops.append(torch.tensor(float(x.numel() * y.numel()), device=x.device))
        return sum(outs), torch.stack(synops, dim=0)

