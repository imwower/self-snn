import argparse
from pathlib import Path

from self_snn.core.workspace import SelfSNN, SelfSNNConfig
from self_snn.utils.viz import save_curve
from self_snn.utils.logging_cn import setup_logger, log_consistency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs/agency_self")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    logger = setup_logger(args.logdir)
    model = SelfSNN(SelfSNNConfig())

    credit_curve = []
    brier_curve = []
    ece_curve = []

    commit_hits = 0
    commit_total = 0
    uncommit_hits = 0
    uncommit_total = 0

    for _ in range(args.epochs):
        out = model(steps=args.steps)
        credit_curve.append(float(out["self_credit"]))
        stats = out.get("consistency_stats", {})
        brier_curve.append(float(stats.get("brier", 0.0)))
        ece_curve.append(float(stats.get("ece", 0.0)))

        commit_state = out["commit_state"]
        act_out = out["act_out"]
        committed = bool(commit_state.get("committed", False))
        success = float(act_out.get("action", 0.0).abs() > 0.01)
        if committed:
            commit_total += 1
            commit_hits += int(success)
        else:
            uncommit_total += 1
            uncommit_hits += int(success)

    figs_dir = Path(args.logdir) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    save_curve(credit_curve, figs_dir / "self_credit.png", ylabel="credit", title="Self Credit")
    save_curve(brier_curve, figs_dir / "self_brier.png", ylabel="brier", title="Self-Consistency Brier")
    save_curve(ece_curve, figs_dir / "self_ece.png", ylabel="ece", title="Self-Consistency ECE")

    # 中文日志摘要与承诺收益对比
    commit_rate = commit_hits / commit_total if commit_total > 0 else 0.0
    uncommit_rate = uncommit_hits / uncommit_total if uncommit_total > 0 else 0.0
    gain = commit_rate - uncommit_rate

    if credit_curve:
        log_consistency(
            logger,
            f"最终信用分={credit_curve[-1]:.3f}, Brier≈{brier_curve[-1]:.3f}, ECE≈{ece_curve[-1]:.3f}; "
            f"承诺期成功率={commit_rate:.3f}, 未承诺期成功率={uncommit_rate:.3f}, 差值={gain:.3f}",
        )

    print(
        f"saved self-consistency figs to {figs_dir} "
        f"(commit_success={commit_rate:.3f}, uncommit_success={uncommit_rate:.3f}, delta={gain:.3f})"
    )


if __name__ == "__main__":
    main()
