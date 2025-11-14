import argparse

from self_snn.core.workspace import SelfSNN, SelfSNNConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs/s0_minimal")
    args = parser.parse_args()

    model = SelfSNN(SelfSNNConfig())
    out = model(steps=300)
    spikes = out["spikes"]
    rate = float(spikes.float().mean() * 1000.0 / 1.0)
    print(f"Ignition eval: mean rate={rate:.3f} Hz")


if __name__ == "__main__":
    main()

