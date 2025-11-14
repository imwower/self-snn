import argparse

from self_snn.core.workspace import SelfSNN, SelfSNNConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    model = SelfSNN(SelfSNNConfig())
    out = model(steps=args.steps)

    router_stats = out["router_stats"]
    moe_ratio = float(out["moe_energy_ratio"])
    topk = router_stats["topk"]

    print("router topk indices:", topk.tolist())
    print(f"estimated MoE energy ratio (masked/dense synops): {moe_ratio:.3f}")


if __name__ == "__main__":
    main()
