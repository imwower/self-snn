from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch

from ..memory.delay_mem import DelayMemory
from ..worldmodel.pred_code import PredictiveCoder
from ..worldmodel.imagination import ImaginationEngine
from .self_model import SelfModel


@dataclass
class IntentionConfig:
    """
    意图生成与效用评估配置。

    Parameters
    ----------
    n_candidates:
        候选目标数量。
    horizon:
        想象 rollout 的步数。
    w_lp, w_empower, w_reward:
        对应 LP_hat / Empower_hat / R_ext_hat 的权重。
    c_energy, c_risk, c_boredom:
        对应能耗 / 风险 / 无聊度的惩罚系数。
    """

    n_candidates: int = 5
    horizon: int = 10
    w_lp: float = 0.45
    w_empower: float = 0.2
    w_reward: float = 0.15
    c_energy: float = 0.1
    c_risk: float = 0.07
    c_boredom: float = 0.03


class IntentionModule:
    """
    意图生成模块：基于 Self-Key 的扰动生成候选目标 g，并结合
    世界模型的误差与想象的估计计算效用 U(g)。
    """

    def __init__(self, config: IntentionConfig) -> None:
        self.config = config

    def _utility(
        self,
        g: torch.Tensor,
        pred: PredictiveCoder,
        imagination: Optional[ImaginationEngine],
        self_key: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算单个候选目标 g 的效用。

        这里用简单启发式近似：
        - LP_hat: 来自预测编码误差的负范数
        - Empower_hat: g 的第 1 维绝对值
        - R_ext_hat: g 的第 2 维
        - Energy/Risk/Boredom: 来自想象模块或 g 的范数 proxy
        """
        # 预测编码误差近似 LP_hat
        _, err = pred(g)
        lp_hat = -err.pow(2).mean()

        empower_hat = g[1].abs()
        r_ext_hat = g[2]

        if imagination is not None:
            h0 = self_key[: pred.config.hidden_dim].float()
            feats = imagination.estimate_utility_terms(g, h0)
            energy = torch.tensor(feats["energy"])
            risk = torch.tensor(feats["risk"])
            boredom = torch.tensor(feats["boredom"])
        else:
            energy = g[3].abs()
            risk = g[4].abs()
            boredom = g[5].abs()

        return (
            self.config.w_lp * lp_hat
            + self.config.w_empower * empower_hat
            + self.config.w_reward * r_ext_hat
            - self.config.c_energy * energy
            - self.config.c_risk * risk
            - self.config.c_boredom * boredom
        )

    def explain_utility(
        self,
        g: torch.Tensor,
        pred: PredictiveCoder,
        imagination: Optional[ImaginationEngine],
        self_key: torch.Tensor,
    ) -> dict:
        """
        对单个候选目标 g 的效用进行分解，返回各组成项及总 U(g)。
        """
        _, err = pred(g)
        lp_hat = -err.pow(2).mean()
        empower_hat = g[1].abs()
        r_ext_hat = g[2]

        if imagination is not None:
            h0 = self_key[: pred.config.hidden_dim].float()
            feats = imagination.estimate_utility_terms(g, h0)
            energy = torch.tensor(feats["energy"])
            risk = torch.tensor(feats["risk"])
            boredom = torch.tensor(feats["boredom"])
        else:
            energy = g[3].abs()
            risk = g[4].abs()
            boredom = g[5].abs()

        u = (
            self.config.w_lp * lp_hat
            + self.config.w_empower * empower_hat
            + self.config.w_reward * r_ext_hat
            - self.config.c_energy * energy
            - self.config.c_risk * risk
            - self.config.c_boredom * boredom
        )
        return {
            "U": u.detach(),
            "LP_hat": lp_hat.detach(),
            "Empower_hat": empower_hat.detach(),
            "R_ext_hat": r_ext_hat.detach(),
            "Energy": energy.detach(),
            "Risk": risk.detach(),
            "Boredom": boredom.detach(),
        }

    def __call__(
        self,
        memory: DelayMemory,
        pred: PredictiveCoder,
        self_model: SelfModel,
        imagination: Optional[ImaginationEngine] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        生成候选目标并计算效用。

        Parameters
        ----------
        memory:
            延迟记忆模块（当前实现未直接使用，为未来扩展预留）。
        pred:
            世界模型的预测编码器。
        self_model:
            自我表征模型，用于提供 Self-Key 与状态。
        imagination:
            想象引擎，用于估计能耗/风险/无聊度等项；可为 None。
        """
        goals: List[torch.Tensor] = []
        utilities: List[torch.Tensor] = []
        base = self_model.key[:6].float()
        for _ in range(self.config.n_candidates):
            noise = 0.1 * torch.randn_like(base)
            g = base + noise
            u = self._utility(g, pred, imagination, self_model.key.float())
            goals.append(g)
            utilities.append(u)
        return goals, torch.stack(utilities, dim=0)
