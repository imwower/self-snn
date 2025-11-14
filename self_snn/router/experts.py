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
        self.register_buffer("usage", torch.zeros(config.num_experts))

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
                # 记录专家使用频率，后续用于剪枝 / 结构调整
                self.usage[idx] += 1.0
                synops.append(torch.tensor(float(x.numel() * y.numel()), device=x.device))
        return sum(outs), torch.stack(synops, dim=0)

    def reset_usage(self) -> None:
        self.usage.zero_()

    def prune(self, min_usage: float = 1.0) -> None:
        for idx, expert in enumerate(self.experts):
            if float(self.usage[idx]) < min_usage:
                expert.weight_mask.zero_()

    def grow(self, n_new: int = 1) -> None:
        for _ in range(n_new):
            new_expert = MaskLinear(self.config.input_dim, self.config.output_dim, bias=True)
            self.experts.append(new_expert)
            self.usage = torch.cat(
                [self.usage, torch.zeros(1, device=self.usage.device, dtype=self.usage.dtype)], dim=0
            )
        self.config.num_experts = len(self.experts)
