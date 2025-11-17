from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


class MaskLinear(nn.Linear):
    """
    线性层带硬掩码：在前向中通过逐元素乘法将未激活权重置零。

    Notes
    -----
    - 对于 MoE 中被完全屏蔽的专家，通常在更高一层（如 MaskedExperts）
      会直接跳过该层的调用，从而实现“零运算”。
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("weight_mask", torch.ones_like(self.weight))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.weight_mask
        return torch.nn.functional.linear(input, w, self.bias)


class MaskConv2d(nn.Conv2d):
    """
    Conv2d 带硬掩码，用于未来卷积型专家。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.register_buffer("weight_mask", torch.ones_like(self.weight))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.weight_mask
        return nn.functional.conv2d(
            input,
            w,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
