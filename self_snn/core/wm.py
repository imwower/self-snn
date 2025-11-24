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
    - 对外仍保持 `forward(spikes)` 接口：接收形状为 [T, N] 的脉冲序列，
      输出长度为 `microcolumns` 的 WM 状态向量，作为上层世界模型 /
      Router 的紧凑输入表征。
    - 内部采用「按神经元平均分组 → 每个分组做短时易化 EMA」的方式，
      近似 θ-γ 分块；不强依赖于 `neurons_per_uCol` 与实际 N 的精确匹配。
    - 同时提供 `write(key, z)` / `read(key)` / `decay()` API，
      供 D-MEM 与世界模型 / Agency 通过键值形式绑定与检索。
    """

    def __init__(self, config: WorkingMemoryConfig) -> None:
        super().__init__()
        self.config = config
        # WM 状态向量：每个 microcolumn 一个槽位
        self.register_buffer("state", torch.zeros(config.microcolumns))
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
            形状为 [microcolumns] 的向量，用作上层世界模型 / Router 的输入。
        """
        assert spikes.dim() == 2, "WorkingMemory 期望输入形状为 [T, N]"
        T, N = spikes.shape
        if T == 0 or N == 0:
            return self.state

        device = spikes.device
        # 确保 WM 状态与输入在同一设备上
        if self.state.device != device:
            self.state = self.state.to(device)

        mc = max(int(self.config.microcolumns), 1)
        spikes_f = spikes.float()

        if mc == 1 or N <= mc:
            # 退化情形：仅有一个或少量分组，直接整体平均
            col_summary = spikes_f.mean(dim=(0, 1)).view(1)
        else:
            # 将 N 个神经元尽量平均划分到 microcolumns 个分组中
            base = N // mc
            rem = N % mc
            split_sizes = []
            for i in range(mc):
                sz = base + (1 if i < rem else 0)
                if sz > 0:
                    split_sizes.append(sz)
            if not split_sizes:
                # 极端情况下（N 非常小），退回到单槽 WM
                col_summary = spikes_f.mean(dim=(0, 1)).view(1)
                mc = 1
            else:
                chunks = torch.split(spikes_f, split_sizes, dim=1)
                # 每个分组：先在时间维度上做平均，再在时间上求整体平均
                col_means = [c.mean(dim=1) for c in chunks]  # 每个 [T]
                col_summary = torch.stack(col_means, dim=0).mean(dim=1)  # [n_groups]

        # 若实际分组数 < microcolumns，用 0 补齐；若更多则截断
        if col_summary.numel() != self.state.numel():
            if col_summary.numel() < self.state.numel():
                pad = torch.zeros(
                    self.state.numel() - col_summary.numel(),
                    device=device,
                    dtype=col_summary.dtype,
                )
                col_summary = torch.cat([col_summary, pad], dim=0)
            else:
                col_summary = col_summary[: self.state.numel()]

        alpha = 1.0 / max(self.config.stf_tau_ms, 1.0)
        self.state = (1.0 - alpha) * self.state + alpha * col_summary
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
