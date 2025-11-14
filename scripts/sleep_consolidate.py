import argparse
from pathlib import Path

import numpy as np
import torch

from self_snn.core.workspace import SelfSNN, SelfSNNConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs/agency_s3")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    model = SelfSNN(SelfSNNConfig())
    optimizer = torch.optim.Adam(model.pred.parameters(), lr=args.lr)
    credits = []

    for epoch in range(args.epochs):
        # 清醒阶段：正常运行，写入 D-MEM
        out = model(steps=args.steps)
        credits.append(float(out["self_credit"]))

        # 睡眠阶段：对已有记忆进行巩固 + 基于 replay 训练世界模型
        model.memory.consolidate()

        key = model.self_model.key
        seq = model.memory.read(key)
        if seq is None:
            continue

        if seq.dim() > 1:
            seq_vec = seq.view(-1)
        else:
            seq_vec = seq

        pred, err = model.pred(seq_vec)
        loss = (err ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    curve_path = logdir / "self_credit_curve.npy"
    np.save(curve_path, np.asarray(credits, dtype="float32"))
    print(f"saved self_credit_curve to {curve_path}")


if __name__ == "__main__":
    main()
