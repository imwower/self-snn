from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn


@dataclass
class MetaConfig:
    conf_window_ms: float = 300.0
    temp_base: float = 0.7


class MetaIntrospector(nn.Module):
    def __init__(self, config: MetaConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, prediction_error: torch.Tensor, salience_gain: torch.Tensor) -> Dict[str, Any]:
        err = prediction_error.abs().mean()
        confidence = torch.exp(-err)
        uncertainty = 1.0 - confidence
        conflict = salience_gain * err
        temp = self.config.temp_base + 0.3 * uncertainty
        return {
            "confidence": confidence,
            "uncertainty": uncertainty,
            "conflict": conflict,
            "temperature": temp,
        }

