from dataclasses import dataclass
from typing import Dict, Any

import torch


@dataclass
class ConsistencyConfig:
    ema_tau: float = 1000.0


class ConsistencyModule:
    def __init__(self, config: ConsistencyConfig) -> None:
        self.config = config
        self._credit = torch.tensor(0.5)

    def __call__(self, commit_state: Dict[str, Any], act_out: Dict[str, Any]) -> torch.Tensor:
        success_signal = float(act_out.get("action", 0.0).abs() > 0.01)
        alpha = 1.0 / max(self.config.ema_tau, 1.0)
        self._credit = (1 - alpha) * self._credit + alpha * success_signal
        return self._credit.detach()

