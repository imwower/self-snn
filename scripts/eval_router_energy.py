import argparse

import torch

from self_snn.router.router import GWRouter, RouterConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs/agency_moe")
    args = parser.parse_args()

    router = GWRouter(RouterConfig())
    wm_state = torch.randn(128)
    mask, stats = router(wm_state)
    print("router topk", stats["topk"], "balance_loss", float(stats["balance_loss"]))


if __name__ == "__main__":
    main()

