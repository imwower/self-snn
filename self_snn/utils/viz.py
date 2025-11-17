from pathlib import Path
from typing import Sequence, Union, Mapping

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


def save_intent_utility_with_components(
    best_utils: Union[torch.Tensor, np.ndarray, Sequence[float]],
    components: Mapping[str, Union[float, torch.Tensor]],
    path: str | Path,
) -> None:
    """
    绘制意图效用曲线 + 分项条形图，输出到 intent_utility.png。

    Parameters
    ----------
    best_utils:
        每个采样步骤上的最佳 U(g) 序列。
    components:
        单个候选目标 g 的效用分解组件，如
        {"LP_hat":..., "Empower_hat":..., "R_ext_hat":..., "Energy":..., "Risk":..., "Boredom":..., "U":...}
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    y_curve = _to_numpy(best_utils)
    x_curve = np.arange(len(y_curve))

    labels = []
    values = []
    for k in ["LP_hat", "Empower_hat", "R_ext_hat", "Energy", "Risk", "Boredom"]:
        if k in components:
            v = components[k]
            if isinstance(v, torch.Tensor):
                v = float(v.detach())
            values.append(v)
            labels.append(k)

    fig, (ax_curve, ax_bar) = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={"height_ratios": [2, 1]})

    # 上：最佳 U(g) 曲线
    ax_curve.plot(x_curve, y_curve, marker="o", linewidth=1.0)
    ax_curve.set_xlabel("epoch")
    ax_curve.set_ylabel("best U(g)")
    ax_curve.set_title("Intent Utility Over Time")

    # 下：分项条形图
    x_bar = np.arange(len(labels))
    ax_bar.bar(x_bar, values)
    ax_bar.set_xticks(x_bar)
    ax_bar.set_xticklabels(labels, rotation=45)
    ax_bar.set_ylabel("value")
    ax_bar.set_title("Utility Components")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

