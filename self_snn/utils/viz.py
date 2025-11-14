from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch


def save_raster(spikes: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t, n = spikes.shape
    fig, ax = plt.subplots(figsize=(6, 4))
    ts, ns = torch.nonzero(spikes, as_tuple=True)
    ax.scatter(ts.cpu(), ns.cpu(), s=1)
    ax.set_xlabel("t")
    ax.set_ylabel("neuron")
    fig.savefig(path)
    plt.close(fig)

