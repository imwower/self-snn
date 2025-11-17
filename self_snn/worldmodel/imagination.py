from dataclasses import dataclass
from typing import Dict, Any

import torch


@dataclass
class ImaginationConfig:
    """
    想象模块配置。

    Parameters
    ----------
    horizon:
        rollout 的步数。
    """

    horizon: int = 10


class ImaginationEngine:
    """
    利用潜在向量 z 与初始状态 h 做简单 rollout，
    并从中估计 LP_hat / Energy / Risk / Boredom 等效用组成项。
    """

    def __init__(self, config: ImaginationConfig) -> None:
        self.config = config

    def rollout(self, z: torch.Tensor, h: torch.Tensor) -> Dict[str, Any]:
        """
        简化 rollout：h_{t+1} = h_t + 0.1 * tanh(z_projected)。

        若 z 与 h 维度不同，则对 z 进行裁剪/补零到 h 的维度。
        """
        if z.dim() == 1:
            z_vec = z
        else:
            z_vec = z.view(-1)
        if z_vec.numel() != h.numel():
            # 将 z 投影到与 h 相同的维度
            if z_vec.numel() > h.numel():
                z_vec = z_vec[: h.numel()]
            else:
                z_vec = torch.nn.functional.pad(z_vec, (0, h.numel() - z_vec.numel()))
        z_vec = z_vec.to(h.device)

        traj = [h]
        cur = h
        for _ in range(self.config.horizon):
            cur = cur + 0.1 * torch.tanh(z_vec)
            traj.append(cur)
        traj_t = torch.stack(traj, dim=0)
        return {"traj": traj_t}

    @torch.no_grad()
    def estimate_utility_terms(self, z: torch.Tensor, h: torch.Tensor) -> Dict[str, float]:
        """
        基于 rollout 轨迹的简单启发式，估计效用组成项：

        - LP_hat: 负的轨迹方差（越稳定越好）
        - Energy: z 的 L2 能量
        - Risk:   轨迹末状态的 L2 作为“偏离原点”的风险 proxy
        - Boredom: z 的 L1 范数的反比（越接近 0 越无聊）
        """
        out = self.rollout(z, h)
        traj = out["traj"]  # [T+1, D]
        # 负方差近似 LP（稳定性）
        lp_hat = float(-traj.var(dim=0).mean().item())
        energy = float(z.pow(2).mean().item())
        risk = float(traj[-1].pow(2).mean().item())
        boredom = float(1.0 / (z.abs().mean().item() + 1e-6))
        return {
            "lp_hat": lp_hat,
            "energy": energy,
            "risk": risk,
            "boredom": boredom,
        }
