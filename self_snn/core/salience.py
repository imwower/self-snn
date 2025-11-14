from dataclasses import dataclass

import torch


@dataclass
class SalienceConfig:
    err_tau_ms: float = 50.0
    gain_max: float = 2.0


class SalienceModule(torch.nn.Module):
    def __init__(self, config: SalienceConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("err_trace", torch.tensor(0.0))

    @torch.no_grad()
    def forward(self, prediction_error: torch.Tensor, dt_ms: float) -> torch.Tensor:
        err_norm = prediction_error.abs().mean()
        alpha = dt_ms / max(self.config.err_tau_ms, 1.0)
        self.err_trace = (1 - alpha) * self.err_trace + alpha * err_norm
        gain = 1.0 + torch.clamp(self.err_trace / (self.err_trace + 1e-6), 0.0, 1.0) * (
            self.config.gain_max - 1.0
        )
        return gain

