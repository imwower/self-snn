import argparse
from pathlib import Path

import torch

from self_snn.core.workspace import SelfSNN, SelfSNNConfig
from self_snn.utils.viz import save_curve, save_heatmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs/agency_figs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    model = SelfSNN(SelfSNNConfig())

    ignition_rates = []
    kappas = []
    energy_curve = []
    moe_ratios = []
    self_credit = []
    best_utils = []
    commit_flags = []
    router_probs_list = []

    for _ in range(args.epochs):
        out = model(steps=args.steps)

        ignition_rates.append(float(out["ignition_rate"]))
        kappas.append(float(out["branching_kappa"]))
        energy_curve.append(float(out["act_out"].get("energy", 0.0)))
        moe_ratios.append(float(out["moe_energy_ratio"]))
        self_credit.append(float(out["self_credit"]))

        utilities = out["utilities"]
        if utilities.numel() > 0:
            best_utils.append(float(utilities.max().detach()))
        else:
            best_utils.append(0.0)

        commit_state = out["commit_state"]
        committed = 1.0 if commit_state.get("committed", False) else 0.0
        commit_flags.append(committed)

        router_probs_list.append(out["router_stats"]["probs"].detach().cpu())

    router_probs = torch.stack(router_probs_list, dim=0)

    outdir = Path(args.logdir) / "figs"
    outdir.mkdir(parents=True, exist_ok=True)

    # 能耗曲线
    save_curve(energy_curve, outdir / "energy_curve.png", ylabel="energy", title="Energy Curve")
    # 自我信用分
    save_curve(self_credit, outdir / "self_credit.png", ylabel="self_credit", title="Self Credit")
    # 意图效用
    save_curve(best_utils, outdir / "intent_utility.png", ylabel="best U(g)", title="Intent Utility")
    # 承诺率曲线
    save_curve(commit_flags, outdir / "commit_rate_curve.png", ylabel="commit(0/1)", title="Commit Rate Curve")
    # 简单 agency 时间线（这里用 self_credit 代表）
    save_curve(self_credit, outdir / "agency_timeline.png", ylabel="self_credit", title="Agency Timeline")
    # Router 使用热力图
    save_heatmap(router_probs, outdir / "router_usage.png", xlabel="expert", ylabel="epoch", title="Router Usage")

    print(f"saved figs to {outdir}")


if __name__ == "__main__":
    main()

