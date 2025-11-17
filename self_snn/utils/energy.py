from dataclasses import dataclass

import torch


@dataclass
class EnergyStats:
    """
    能耗统计结果。

    Attributes
    ----------
    synops_masked:
        在条件计算（MoE）下实际执行的突触运算量估计。
    synops_dense:
        对应密集前向（所有专家都激活）时的突触运算量估计。
    """

    synops_masked: float = 0.0
    synops_dense: float = 0.0

    @property
    def moe_ratio(self) -> float:
        """
        条件计算能耗比（masked/dense synops）。
        """
        if self.synops_dense <= 0.0:
            return 1.0
        return self.synops_masked / self.synops_dense


def count_spikes(spikes: torch.Tensor) -> float:
    """
    统计脉冲总数。
    """
    return float(spikes.sum().item())


def compute_moe_energy(hidden: torch.Tensor, synops_per_expert: torch.Tensor, num_experts: int) -> EnergyStats:
    """
    依据隐藏向量维度与每个专家的 synops 估计条件计算能耗。

    Parameters
    ----------
    hidden:
        MoE 输入隐状态向量。
    synops_per_expert:
        每个专家实际执行的 synops 估计（通常由 MaskedExperts 返回）。
    num_experts:
        专家数量 M。
    """
    synops_masked = float(synops_per_expert.sum().item())
    per_expert_cost = float(hidden.numel() * hidden.numel())
    synops_dense = per_expert_cost * float(num_experts)
    return EnergyStats(synops_masked=synops_masked, synops_dense=synops_dense)
