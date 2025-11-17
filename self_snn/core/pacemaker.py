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
        # 目标发放率作为可调参数，用于近临界调参（根据 κ 在线微调）
        self.register_buffer("target_rate", torch.tensor(config.target_rate_hz, dtype=torch.float32))

    @torch.no_grad()
    def forward(self, T: int) -> torch.Tensor:
        dt = self.config.dt_ms / 1000.0
        sigma = self.config.ou_sigma
        theta = 1.0
        rate = float(self.target_rate)
        spikes = []
        x = self.state
        for _ in range(T):
            # OU 噪声驱动的慢变量 x，但仅作为发放率的小幅调制项
            noise = torch.randn_like(x) * sigma * (dt ** 0.5)
            x = x + (-theta * x) * dt + noise
            # 把 x 压缩到 [-0.1, 0.1] 的窄范围内，避免远离目标发放率
            mod = 0.1 * torch.tanh(x)
            p_base = rate * dt
            p_spike = torch.clamp(p_base * (1.0 + mod), 0.0, 1.0)
            s = torch.bernoulli(p_spike)
            spikes.append(s)
        out = torch.stack(spikes, dim=0)
        self.state = x.detach()
        return out

    @torch.no_grad()
    def adapt_to_branching(self, kappa: torch.Tensor, target: float = 1.0, lr: float = 0.01) -> None:
        """
        简化的近临界调参：根据在线估计的分支系数 κ 微调自发发放率。

        - 若 κ > target，则略微降低 target_rate；
        - 若 κ < target，则略微提高 target_rate。
        """
        kappa_val = float(kappa.detach())
        err = kappa_val - target
        # 线性近似的乘性更新，并限制在合理范围
        self.target_rate *= torch.tensor(1.0 - lr * err, dtype=self.target_rate.dtype, device=self.target_rate.device)
        self.target_rate.clamp_(0.1, 20.0)
