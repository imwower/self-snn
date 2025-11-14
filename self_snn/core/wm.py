from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class WorkingMemoryConfig:
    microcolumns: int = 3
    neurons_per_uCol: tuple[int, int] = (80, 20)
    stf_tau_ms: float = 200.0


class WorkingMemory(nn.Module):
    def __init__(self, config: WorkingMemoryConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("state", torch.zeros(1))

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        # Aggregate spikes over neurons into a single scalar per time step,
        # then average over the time window for a compact WM state.
        spikes_flat = spikes.float().mean(dim=1)
        summary = spikes_flat.mean().unsqueeze(0)
        alpha = 1.0 / max(self.config.stf_tau_ms, 1.0)
        self.state = (1 - alpha) * self.state + alpha * summary
        return self.state
