from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from ..utils.masks import MaskLinear


@dataclass
class ExpertsConfig:
    """
    专家网络配置。

    Parameters
    ----------
    input_dim:
        每个专家输入维度。
    output_dim:
        每个专家输出维度。
    num_experts:
        专家数量 M。
    """

    input_dim: int = 128
    output_dim: int = 128
    num_experts: int = 16


class MaskedExperts(nn.Module):
    """
    带 Top-K 掩码的专家集合。

    Notes
    -----
    - 通过 `mask[idx]` 是否为 1 决定是否执行第 idx 个专家的前向计算；
      未选中的专家直接输出零张量，从而在前后向中都实现“零运算”。
    - 返回的 synops 仅对被选专家给出非零估计：约为 input_dim * output_dim。
    """

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
                # 未选专家：完全跳过线性层调用，前后向零 FLOPs
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
