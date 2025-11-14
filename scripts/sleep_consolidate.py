import argparse
from pathlib import Path

import numpy as np

from self_snn.core.workspace import SelfSNN, SelfSNNConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs/agency_s3")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    model = SelfSNN(SelfSNNConfig())
    credits = []

    for epoch in range(args.epochs):
        out = model(steps=args.steps)
        credits.append(float(out["self_credit"]))
        model.memory.consolidate()

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    curve_path = logdir / "self_credit_curve.npy"
    np.save(curve_path, np.asarray(credits, dtype="float32"))
    print(f"saved self_credit_curve to {curve_path}")


if __name__ == "__main__":
    main()

