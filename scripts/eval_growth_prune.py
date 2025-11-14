import argparse

import torch

from self_snn.router.router import GWRouter, RouterConfig
from self_snn.router.experts import MaskedExperts, ExpertsConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    router = GWRouter(RouterConfig(num_experts=8, k=2))
    experts = MaskedExperts(ExpertsConfig(input_dim=16, output_dim=16, num_experts=8))

    x = torch.randn(16)
    for _ in range(args.steps):
        wm_state = torch.randn(16)
        mask, _ = router(wm_state)
        _, synops = experts(x, mask)

    print("usage before prune:", experts.usage.tolist())
    experts.prune(min_usage=args.steps * 0.1)
    experts.grow(n_new=2)
    print("num experts after grow:", experts.config.num_experts)


if __name__ == "__main__":
    main()

