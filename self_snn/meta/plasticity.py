from dataclasses import dataclass

import torch


@dataclass
class STDPConfig:
    a_plus: float = 0.01
    a_minus: float = 0.012
    tau_plus: float = 20.0
    tau_minus: float = 20.0


def stdp_update(weights: torch.Tensor, pre: torch.Tensor, post: torch.Tensor, config: STDPConfig) -> torch.Tensor:
    dw = torch.zeros_like(weights)
    dw += config.a_plus * torch.outer(pre, post)
    dw -= config.a_minus * torch.outer(post, pre)
    return weights + dw

