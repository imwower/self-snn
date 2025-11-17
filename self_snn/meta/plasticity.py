from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class STDPConfig:
    """
    STDP 配置参数。

    Attributes
    ----------
    a_plus:
        pre→post 顺序发放时的权重增益系数。
    a_minus:
        post→pre 反向发放时的权重减益系数。
    tau_plus:
        pre 侧资格迹时间常数（步数单位，假定 dt=1）。
    tau_minus:
        post 侧资格迹时间常数。
    """

    a_plus: float = 0.01
    a_minus: float = 0.012
    tau_plus: float = 20.0
    tau_minus: float = 20.0


@dataclass
class STDPState:
    """
    STDP 三因子学习的内部状态（资格迹）。

    Attributes
    ----------
    pre_trace:
        pre 侧资格迹，形状与 pre 向量相同。
    post_trace:
        post 侧资格迹，形状与 post 向量相同。
    """

    pre_trace: torch.Tensor
    post_trace: torch.Tensor


def stdp_update(weights: torch.Tensor, pre: torch.Tensor, post: torch.Tensor, config: STDPConfig) -> torch.Tensor:
    """
    兼容旧接口的简单 STDP 更新（两因子，无资格迹与第三因子）。

    Notes
    -----
    - 该函数保持原有签名不变，方便已有代码直接调用。
    - 完整三因子版本请使用 `stdp_3factor`。
    """
    dw = torch.zeros_like(weights)
    dw += config.a_plus * torch.outer(pre, post)
    dw -= config.a_minus * torch.outer(post, pre)
    return weights + dw


def stdp_3factor(
    weights: torch.Tensor,
    pre_spikes: torch.Tensor,
    post_spikes: torch.Tensor,
    state: STDPState,
    config: STDPConfig,
    third_factor: torch.Tensor,
    dt: float = 1.0,
) -> Tuple[torch.Tensor, STDPState]:
    """
    三因子 STDP 更新（资格迹 + 第三因子）。

    Parameters
    ----------
    weights:
        突触权重矩阵，形状 [N_pre, N_post]。
    pre_spikes:
        pre 侧脉冲向量，形状 [N_pre]。
    post_spikes:
        post 侧脉冲向量，形状 [N_post]。
    state:
        STDPState，包含 pre_trace / post_trace。
    config:
        STDPConfig，包含 A+/A- 与时间常数。
    third_factor:
        第三因子调制信号（如 RPE / Surprise / Empowerment 等），
        可为标量或与 weights 可广播的张量。
    dt:
        时间步长（默认为 1.0，用于资格迹衰减）。

    Returns
    -------
    new_weights, new_state:
        更新后的权重与资格迹状态。
    """
    # 计算资格迹衰减因子（标量）
    decay_plus = torch.exp(torch.tensor(-dt / max(config.tau_plus, 1.0), dtype=weights.dtype, device=weights.device))
    decay_minus = torch.exp(
        torch.tensor(-dt / max(config.tau_minus, 1.0), dtype=weights.dtype, device=weights.device)
    )

    pre_trace = state.pre_trace * decay_plus + pre_spikes
    post_trace = state.post_trace * decay_minus + post_spikes

    # 使用资格迹构造 STDP 权重更新：pre_trace × post_spikes 和 post_trace × pre_spikes
    dw_plus = config.a_plus * torch.outer(pre_trace, post_spikes)
    dw_minus = config.a_minus * torch.outer(pre_spikes, post_trace)

    # 第三因子调制（可为标量或可广播张量）
    if not isinstance(third_factor, torch.Tensor):
        third = torch.tensor(third_factor, dtype=weights.dtype, device=weights.device)
    else:
        third = third_factor.to(weights.device, dtype=weights.dtype)

    dw = (dw_plus - dw_minus) * third
    new_weights = weights + dw
    new_state = STDPState(pre_trace=pre_trace, post_trace=post_trace)
    return new_weights, new_state
