from dataclasses import dataclass
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn


@dataclass
class RouterConfig:
    num_experts: int = 16
    k: int = 2
    z_loss: float = 1e-3
    usage_ema_tau: float = 1000.0


class GWRouter(nn.Module):
    def __init__(self, config: RouterConfig) -> None:
        super().__init__()
        self.config = config
        self.prototypes = nn.Parameter(torch.randn(config.num_experts, 1))
        # 路由使用频率的指数滑动平均，用于负载均衡正则
        self.register_buffer("usage_ema", torch.zeros(config.num_experts))
        self._step = 0

    def forward(self, wm_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x = wm_state.mean().unsqueeze(0)
        sim = -(self.prototypes - x).pow(2).squeeze(-1)
        probs = torch.softmax(sim, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k=self.config.k, dim=-1)
        mask = torch.zeros_like(probs)
        mask[topk_idx] = 1.0

        # 负载均衡：使用 usage_ema 逼近均匀分布，避免“热专家”
        self._step += 1
        alpha = 1.0 / max(float(self.config.usage_ema_tau), 1.0)
        self.usage_ema = (1 - alpha) * self.usage_ema + alpha * mask.detach()
        target = torch.full_like(self.usage_ema, 1.0 / self.config.num_experts)
        usage_balance = (self.usage_ema - target).pow(2).mean()
        balance_loss = usage_balance * self.config.z_loss

        stats = {
            "probs": probs.detach(),
            "balance_loss": balance_loss,
            "topk": topk_idx.detach(),
            "usage_ema": self.usage_ema.detach(),
        }
        return mask, stats
