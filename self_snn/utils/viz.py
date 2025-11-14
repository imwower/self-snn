from pathlib import Path
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(x: Union[torch.Tensor, np.ndarray, Sequence[float]]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(list(x), dtype="float32")


def save_raster(spikes: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t, n = spikes.shape
    fig, ax = plt.subplots(figsize=(6, 4))
    ts, ns = torch.nonzero(spikes, as_tuple=True)
    ax.scatter(ts.cpu(), ns.cpu(), s=1)
    ax.set_xlabel("t")
    ax.set_ylabel("neuron")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_curve(
    values: Union[torch.Tensor, np.ndarray, Sequence[float]],
    path: str | Path,
    xlabel: str = "epoch",
    ylabel: str = "value",
    title: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = _to_numpy(values)
    x = np.arange(len(y))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o", linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_heatmap(
    matrix: Union[torch.Tensor, np.ndarray],
    path: str | Path,
    xlabel: str = "expert",
    ylabel: str = "step",
    title: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mat = _to_numpy(matrix)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mat, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

