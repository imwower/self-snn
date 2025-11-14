import argparse
from pathlib import Path

import torch
import yaml

from self_snn.core.workspace import SelfSNN, SelfSNNConfig
from self_snn.utils.logging_cn import setup_logger, log_pacemaker, log_self


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--logdir", type=str, default="runs/debug")
    parser.add_argument("--duration", type=int, default=180)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("backend", {}).get("device", "cpu"))

    model = SelfSNN(SelfSNNConfig(device=str(device))).to(device)
    logger = setup_logger(args.logdir)

    for epoch in range(3):
        out = model(steps=args.duration)
        log_pacemaker(logger, "θ/γ 节律=..., 噪声σ=... 已启动")
        log_self(logger, model.self_model.report())
        loss = out["prediction_error"].abs().mean()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.data -= 1e-4 * p.grad
                p.grad.zero_()


if __name__ == "__main__":
    main()

