import argparse

import torch

from self_snn.core.workspace import SelfSNN, SelfSNNConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs/agency_s1")
    args = parser.parse_args()

    model = SelfSNN(SelfSNNConfig())
    key = model.self_model.key
    seq = torch.randn(20, 8)
    model.memory.write(key, seq, delay=5)
    recalled = model.memory.read(key)
    replayed = model.memory.replay(key)
    print("memory hit", recalled is not None, "replay shape", replayed.shape if replayed is not None else None)


if __name__ == "__main__":
    main()

