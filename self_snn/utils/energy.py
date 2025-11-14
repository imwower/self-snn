from dataclasses import dataclass

import torch


@dataclass
class EnergyStats:
    synops: float = 0.0
    spikes: float = 0.0


def count_spikes(spikes: torch.Tensor) -> float:
    return float(spikes.sum().item())

