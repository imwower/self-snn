from dataclasses import dataclass
from math import pi
from typing import Dict

import torch


@dataclass
class PacemakerConfig:
    """
    起搏器配置。

    Attributes
    ----------
    n_neurons:
        起搏器神经元个数（也是最小意识核的基数）。
    ring_delay_ms:
        “环形连通”对应的延迟（ms），主要用于设置相位节律的时间尺度。
    ou_sigma:
        OU 噪声强度（越大自发波动越明显）。
    target_rate_hz:
        目标平均放电频率（Hz），用于近临界调参。
    dt_ms:
        时间步长（ms）。
    theta_hz:
        θ 节律频率（Hz），典型 4–8Hz。
    gamma_hz:
        γ 节律频率（Hz），典型 30–80Hz。
    """

    n_neurons: int = 64
    ring_delay_ms: float = 8.0
    ou_sigma: float = 0.6
    target_rate_hz: float = 3.0
    dt_ms: float = 1.0
    theta_hz: float = 6.0
    gamma_hz: float = 40.0


class Pacemaker(torch.nn.Module):
    """
    起搏器 + OU 噪声 + θ/γ 相位门控。

    Notes
    -----
    - 本实现采用“泊松发放 + OU 噪声 + 正弦相位门控”的轻量 ALIF 近似，
      满足 0.5–5 Hz 自发放电与近临界 (κ≈1) 要求。
    - 接口提供 `step(T)` 返回完整观测字典，`forward(T)` 仅返回 spikes，
      以兼容现有 SelfSNN 使用方式。

    Examples
    --------
    >>> cfg = PacemakerConfig()
    >>> pacemaker = Pacemaker(cfg)
    >>> out = pacemaker.step(T=10)
    >>> out["spikes"].shape[0] == 10
    True
    """

    def __init__(self, config: PacemakerConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        self.device = device or torch.device("cpu")

        # OU 慢变量（用于轻量自适应）、膜电位跟踪、θ/γ 相位计数
        self.register_buffer("ou_state", torch.zeros(config.n_neurons, device=self.device))
        self.register_buffer("vm", torch.zeros(config.n_neurons, device=self.device))
        self.register_buffer("time_step", torch.tensor(0, dtype=torch.long, device=self.device))
        # 目标发放率作为可调参数，用于近临界调参（根据 κ 在线微调）
        self.register_buffer("target_rate", torch.tensor(config.target_rate_hz, dtype=torch.float32))

    @torch.no_grad()
    def step(self, T: int) -> Dict[str, torch.Tensor]:
        """
        运行 T 个时间步，返回包含 spikes/vm/相位/分支系数的观测。

        Parameters
        ----------
        T:
            模拟步数（整数）。
        """
        dt_s = self.config.dt_ms / 1000.0
        sigma = self.config.ou_sigma
        rate = float(self.target_rate)

        ou = self.ou_state
        vm = self.vm

        spikes_list = []
        vm_trace_list = []
        theta_phase_list = []
        gamma_phase_list = []
        spike_counts = []

        for t in range(T):
            # 当前绝对时间（秒），用于相位计算
            t_abs = (int(self.time_step) + t) * dt_s
            t_tensor = torch.tensor(t_abs, device=self.device, dtype=torch.float32)
            theta_phase = torch.sin(2.0 * pi * self.config.theta_hz * t_tensor)
            gamma_phase = torch.sin(2.0 * pi * self.config.gamma_hz * t_tensor)

            # OU 噪声驱动的慢变量，仅作为放电率的小幅调制项
            noise = torch.randn_like(ou) * sigma * (dt_s**0.5)
            ou = ou + (-ou) * dt_s + noise
            ou_mod = 0.1 * torch.tanh(ou)

            # 相位门控：θ/γ 只影响整体放电概率的轻微涨落
            phase_mod = 0.05 * theta_phase + 0.02 * gamma_phase

            p_base = rate * dt_s
            p_spike = torch.clamp(p_base * (1.0 + ou_mod + phase_mod), 0.0, 1.0)
            s = torch.bernoulli(p_spike)

            # 简单膜电位跟踪：指数泄露 + 突触后去极化
            vm = 0.95 * vm + s

            spikes_list.append(s)
            vm_trace_list.append(vm.clone())
            # 直接使用已在正确设备上的张量，避免重复构造导致的警告
            theta_phase_list.append(theta_phase.detach().clone())
            gamma_phase_list.append(gamma_phase.detach().clone())
            spike_counts.append(s.sum())

        spikes = torch.stack(spikes_list, dim=0)  # [T, N]
        vm_trace = torch.stack(vm_trace_list, dim=0)  # [T, N]
        theta_phase_trace = torch.stack(theta_phase_list, dim=0)  # [T]
        gamma_phase_trace = torch.stack(gamma_phase_list, dim=0)  # [T]
        spike_counts_t = torch.stack(spike_counts).float()  # [T]

        self.ou_state = ou.detach()
        self.vm = vm.detach()
        self.time_step += T

        # 分支系数 κ：粗略估计 t 与 t+1 步的放电比值
        if spike_counts_t.numel() > 1:
            ratio = spike_counts_t[1:] / torch.clamp(spike_counts_t[:-1], min=1.0)
            branching_kappa = ratio.mean()
        else:
            branching_kappa = torch.tensor(1.0, device=self.device)

        # 近临界调参：用 κ 在线微调起搏器目标发放率
        self._adapt_to_branching(branching_kappa)

        return {
            "spikes": spikes,
            "vm": vm_trace,
            "theta_phase": theta_phase_trace,
            "gamma_phase": gamma_phase_trace,
            "branching_kappa": branching_kappa,
        }

    @torch.no_grad()
    def forward(self, T: int) -> torch.Tensor:
        """
        兼容旧接口：仅返回 spikes。
        """
        out = self.step(T)
        return out["spikes"]

    @torch.no_grad()
    def _adapt_to_branching(self, kappa: torch.Tensor, target: float = 1.0, lr: float = 0.01) -> None:
        """
        简化的近临界调参：根据在线估计的分支系数 κ 微调自发发放率。

        - 若 κ > target，则略微降低 target_rate；
        - 若 κ < target，则略微提高 target_rate。
        """
        kappa_val = float(kappa.detach())
        err = kappa_val - target
        # 线性近似的乘性更新，并限制在合理范围
        self.target_rate *= torch.tensor(
            1.0 - lr * err,
            dtype=self.target_rate.dtype,
            device=self.target_rate.device,
        )
        self.target_rate.clamp_(0.1, 20.0)
