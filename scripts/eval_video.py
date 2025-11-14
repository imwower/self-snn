import argparse

import torch

from self_snn.core.workspace import SelfSNN, SelfSNNConfig
from self_snn.utils.encoders import encode_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs/agency_s3")
    args = parser.parse_args()

    model = SelfSNN(SelfSNNConfig())
    frames = torch.randn(10, 1, 16, 16)
    emb = encode_video(frames)
    out = model({"drive": emb.unsqueeze(0).repeat(100, 1)}, steps=100)
    print("S3 video eval self_credit", float(out["self_credit"]))


if __name__ == "__main__":
    main()

