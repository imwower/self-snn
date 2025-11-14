import argparse

from self_snn.core.workspace import SelfSNN, SelfSNNConfig
from self_snn.utils.encoders import encode_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs/agency_s2")
    args = parser.parse_args()

    model = SelfSNN(SelfSNNConfig())
    emb = encode_text(["这", "是", "一句", "完整", "的", "句子"])
    out = model({"drive": emb.unsqueeze(0).repeat(100, 1)}, steps=100)
    print("S2 sentence eval self_credit", float(out["self_credit"]))


if __name__ == "__main__":
    main()

