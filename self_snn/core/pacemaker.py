from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class PacemakerConfig:
    n_neurons: int = 64
    ring_delay_ms: float = 8.0
    ou_sigma: float = 0.6
    target_rate_hz: float = 3.0
    dt_ms: float = 1.0


class Pacemaker(torch.nn.Module):
    def __init__(self, config: PacemakerConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        self.device = device or torch.device("cpu")
        self.register_buffer("state", torch.zeros(config.n_neurons, device=self.device))

    @torch.no_grad()
    def forward(self, T: int) -> torch.Tensor:
        dt = self.config.dt_ms / 1000.0
        sigma = self.config.ou_sigma
        theta = 1.0
        rate = self.config.target_rate_hz
        spikes = []
        x = self.state
        for _ in range(T):
            noise = torch.randn_like(x) * sigma * (dt ** 0.5)
            x = x + (-theta * x) * dt + noise
            p_spike = torch.clamp(rate * dt + x, 0.0, 1.0)
            s = torch.bernoulli(p_spike)
            spikes.append(s)
        out = torch.stack(spikes, dim=0)
        self.state = x.detach()
        return out

