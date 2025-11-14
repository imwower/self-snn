from dataclasses import dataclass
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn


@dataclass
class RouterConfig:
    num_experts: int = 16
    k: int = 2
    z_loss: float = 1e-3


class GWRouter(nn.Module):
    def __init__(self, config: RouterConfig) -> None:
        super().__init__()
        self.config = config
        self.prototypes = nn.Parameter(torch.randn(config.num_experts, 1))

    def forward(self, wm_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x = wm_state.mean().unsqueeze(0)
        sim = -(self.prototypes - x).pow(2).squeeze(-1)
        probs = torch.softmax(sim, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k=self.config.k, dim=-1)
        mask = torch.zeros_like(probs)
        mask[topk_idx] = 1.0
        balance_loss = (probs.mean() - 1.0 / self.config.num_experts).pow(2) * self.config.z_loss
        stats = {
            "probs": probs.detach(),
            "balance_loss": balance_loss,
            "topk": topk_idx.detach(),
        }
        return mask, stats

