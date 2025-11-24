from dataclasses import dataclass
from typing import Dict, Any

import torch


@dataclass
class ConsistencyConfig:
    """
    自我一致性（Brier/ECE/信用分）配置。
    """

    ema_tau: float = 1000.0
    ema_tau_brier: float = 500.0
    ema_tau_ece: float = 500.0


class ConsistencyModule:
    """
    基于承诺概率与结果的自我一致性评估。

    - 维护：
        - 自我信用分 credit：类似成功率的 EMA；
        - Brier 分数：概率预测的均方误差；
        - ECE 近似：|p - y| 的 EMA。
    """

    def __init__(self, config: ConsistencyConfig) -> None:
        self.config = config
        self._credit = torch.tensor(0.5)
        self._brier = torch.tensor(0.25)
        self._ece = torch.tensor(0.5)

    def __call__(self, commit_state: Dict[str, Any], act_out: Dict[str, Any]) -> torch.Tensor:
        # 成功信号：将「是否实际承诺执行」视作观测标签，
        # 用于校准承诺概率的自洽性（Brier/ECE）。
        committed = bool(commit_state.get("committed", False))
        y = torch.tensor(1.0 if committed else 0.0, dtype=torch.float32)

        # 承诺概率估计（若无则视为 0.5）
        prob = commit_state.get("prob", torch.tensor(0.5))
        if not isinstance(prob, torch.Tensor):
            prob = torch.tensor(prob, dtype=torch.float32)
        prob = prob.to(dtype=torch.float32)

        # 自我信用分（EMA）
        alpha = 1.0 / max(self.config.ema_tau, 1.0)
        self._credit = (1 - alpha) * self._credit + alpha * y

        # Brier 与 ECE 的 EMA
        alpha_brier = 1.0 / max(self.config.ema_tau_brier, 1.0)
        alpha_ece = 1.0 / max(self.config.ema_tau_ece, 1.0)
        brier_t = (prob - y) ** 2
        ece_t = (prob - y).abs()
        self._brier = (1 - alpha_brier) * self._brier + alpha_brier * brier_t
        self._ece = (1 - alpha_ece) * self._ece + alpha_ece * ece_t

        return self._credit.detach()

    def stats(self) -> Dict[str, torch.Tensor]:
        """
        返回当前自我一致性统计量，供上层记录或可视化。
        """
        return {
            "credit": self._credit.detach(),
            "brier": self._brier.detach(),
            "ece": self._ece.detach(),
        }
