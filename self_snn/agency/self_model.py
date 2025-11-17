from dataclasses import dataclass
from typing import Dict, Any

import torch


@dataclass
class SelfModelConfig:
    key_dim: int = 64


class SelfModel:
    def __init__(self, config: SelfModelConfig) -> None:
        self.config = config
        self.key = torch.sign(torch.randn(config.key_dim))
        self.state = {
            "confidence": 0.5,
            "energy": 0.0,
            "ability": 0.5,
            "risk": 0.0,
        }

    def update_state(self, meta: Dict[str, Any], act_out: Dict[str, Any]) -> None:
        conf_raw = meta.get("confidence", 0.0)
        unc_raw = meta.get("uncertainty", 0.0)
        if isinstance(conf_raw, torch.Tensor):
            conf = float(conf_raw.detach())
        else:
            conf = float(conf_raw)
        if isinstance(unc_raw, torch.Tensor):
            risk = float(unc_raw.detach())
        else:
            risk = float(unc_raw)
        energy = float(act_out.get("energy", 0.0))
        alpha = 0.01
        self.state["confidence"] = (1 - alpha) * self.state["confidence"] + alpha * conf
        self.state["energy"] = (1 - alpha) * self.state["energy"] + alpha * energy
        self.state["risk"] = (1 - alpha) * self.state["risk"] + alpha * risk

    def report(self) -> str:
        return (
            f"置信={self.state['confidence']:.3f} "
            f"能耗={self.state['energy']:.3f} "
            f"能力={self.state['ability']:.3f} "
            f"风险={self.state['risk']:.3f}"
        )
