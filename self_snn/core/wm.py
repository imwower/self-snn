from dataclasses import dataclass
from typing import Dict, Hashable, Optional

import torch
import torch.nn as nn


@dataclass
class WorkingMemoryConfig:
    """
    工作记忆（WM）配置。

    Parameters
    ----------
    microcolumns:
        微柱数量，用于概念上划分 θ-γ 分块（当前实现中不显式使用）。
    neurons_per_uCol:
        每个微柱内的神经元数（exc / inh），保持与 README 一致的接口。
    stf_tau_ms:
        短时易化（short-term facilitation）的时间常数（ms），用于 WM 状态的 EMA。
    """

    microcolumns: int = 3
    neurons_per_uCol: tuple[int, int] = (80, 20)
    stf_tau_ms: float = 200.0


class WorkingMemory(nn.Module):
    """
    θ-γ 分块短时工作记忆。

    Notes
    -----
    - 对外仍保持 `forward(spikes)` 接口：接收 [T, N] 的脉冲序列，
      输出一个紧凑的 WM 状态向量（目前为标量，占位）。
    - 同时提供 `write(key, z)` / `read(key)` / `decay()` API，
      供 D-MEM 与世界模型 / Agency 通过键值形式绑定与检索。
    """

    def __init__(self, config: WorkingMemoryConfig) -> None:
        super().__init__()
        self.config = config
        # 全局 WM 状态（用作 SelfSNN 中的紧凑表征）
        self.register_buffer("state", torch.zeros(1))
        # 键值形式的 WM 存储（概念上的 θ-γ 分块绑定）
        self._kv_store: Dict[Hashable, torch.Tensor] = {}

    @torch.no_grad()
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        将时间-神经元维度的脉冲聚合为一个紧凑 WM 状态。

        Parameters
        ----------
        spikes:
            形状为 [T, N] 的脉冲张量。

        Returns
        -------
        state:
            形状为 [1] 的标量状态，用作上层世界模型 / Router 的输入。
        """
        # 简化实现：对神经元与时间维度分别求均值，
        # 再用 stf_tau_ms 做指数滑动平均。
        spikes_flat = spikes.float().mean(dim=1)  # [T]
        summary = spikes_flat.mean().unsqueeze(0)  # [1]
        alpha = 1.0 / max(self.config.stf_tau_ms, 1.0)
        self.state = (1 - alpha) * self.state + alpha * summary
        return self.state

    @torch.no_grad()
    def write(self, key: torch.Tensor, z: torch.Tensor) -> None:
        """
        写入一个键值对（工作记忆槽）。

        Parameters
        ----------
        key:
            任意维度的张量，将会被二值化后映射为 hashable key。
        z:
            要存储的向量（通常为 WM 或世界模型隐状态）。
        """
        tkey = tuple((key.detach().cpu().flatten() > 0).int().tolist())
        self._kv_store[tkey] = z.detach().clone()

    @torch.no_grad()
    def read(self, key: torch.Tensor) -> Optional[torch.Tensor]:
        """
        按键读取工作记忆中的值。

        若键不存在，返回 None。
        """
        tkey = tuple((key.detach().cpu().flatten() > 0).int().tolist())
        return self._kv_store.get(tkey)

    @torch.no_grad()
    def decay(self, factor: float = 0.95) -> None:
        """
        全局衰减 WM 状态，用于模拟短时保持的自然消退。
        """
        self.state *= factor
