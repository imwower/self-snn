from dataclasses import dataclass
from typing import Tuple, Dict

import torch
import torch.nn as nn


@dataclass
class PredictiveCoderConfig:
    """
    预测编码模块配置。

    Parameters
    ----------
    hidden_dim:
        潜在状态与预测向量的维度。
    """

    hidden_dim: int = 128


class PredictiveCoder(nn.Module):
    """
    简化的 Spiking 预测编码模块。

    Notes
    -----
    - 内部维护一个隐状态 `h`，每次调用 `forward(z)`：
        h_{t+1} = f(h_t + z_t)
        x̂_{t+1} = h_{t+1}
        ε_{t+1} = x̂_{t+1} - z_t
    - 提供 `rollout(z, horizon)` 用于前瞻想象。
    """

    def __init__(self, config: PredictiveCoderConfig) -> None:
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.register_buffer("hidden", torch.zeros(config.hidden_dim))

    def _prepare_input(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            x = z
        else:
            x = z.view(-1)
        if x.numel() != self.config.hidden_dim:
            x = nn.functional.pad(x, (0, max(self.config.hidden_dim - x.numel(), 0)))
            x = x[: self.config.hidden_dim]
        return x

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单步预测编码：z_t -> (x̂_{t+1}, ε_{t+1}).
        """
        z_vec = self._prepare_input(z)
        h_prev = self.hidden
        h_new = self.net(h_prev + z_vec)
        self.hidden = h_new.detach()
        pred = h_new
        err = pred - z_vec
        return pred, err

    @torch.no_grad()
    def rollout(self, z: torch.Tensor, horizon: int) -> Dict[str, torch.Tensor]:
        """
        使用固定输入 z 进行前瞻 rollout。

        Returns
        -------
        dict:
            - traj: [T+1, hidden_dim] 的隐藏状态轨迹
            - err:  [T, hidden_dim] 的误差轨迹
        """
        z_vec = self._prepare_input(z)
        h = self.hidden.clone()
        traj = [h]
        errs = []
        for _ in range(horizon):
            h = self.net(h + z_vec)
            traj.append(h)
            errs.append(h - z_vec)
        traj_t = torch.stack(traj, dim=0)
        err_t = torch.stack(errs, dim=0)
        return {"traj": traj_t, "err": err_t}
