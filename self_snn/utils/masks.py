import torch
import torch.nn as nn


class MaskLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("weight_mask", torch.ones_like(self.weight))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.weight_mask
        return torch.nn.functional.linear(input, w, self.bias)

