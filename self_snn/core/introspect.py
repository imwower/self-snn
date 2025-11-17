from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn


@dataclass
class MetaConfig:
    conf_window_ms: float = 300.0
    temp_base: float = 0.7


class MetaIntrospector(nn.Module):
    def __init__(self, config: MetaConfig) -> None:
        super().__init__()
        self.config = config
        # 多时间尺度 EMA：这里只实现一个“慢窗口”EMA，结合瞬时值一起输出
        self.register_buffer("conf_ema", torch.tensor(0.5))
        self.register_buffer("uncert_ema", torch.tensor(0.5))

    def forward(self, prediction_error: torch.Tensor, salience_gain: torch.Tensor) -> Dict[str, Any]:
        err = prediction_error.abs().mean()
        # 瞬时置信/不确定
        confidence_inst = torch.exp(-err)
        uncertainty_inst = 1.0 - confidence_inst

        # 根据 conf_window_ms 近似一个步长上的 EMA 系数
        alpha = 1.0 / max(self.config.conf_window_ms, 1.0)
        self.conf_ema = (1.0 - alpha) * self.conf_ema + alpha * confidence_inst
        self.uncert_ema = (1.0 - alpha) * self.uncert_ema + alpha * uncertainty_inst

        conflict = salience_gain * err
        temp = self.config.temp_base + 0.3 * self.uncert_ema
        return {
            # 慢时间尺度 EMA（供 Self-Model/日志与 TB 使用）
            "confidence": self.conf_ema.detach(),
            "uncertainty": self.uncert_ema.detach(),
            # 瞬时值（便于未来 fine-grained 分析）
            "confidence_inst": confidence_inst.detach(),
            "uncertainty_inst": uncertainty_inst.detach(),
            "conflict": conflict,
            "temperature": temp,
        }
