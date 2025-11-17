import argparse
from pathlib import Path

from self_snn.core.workspace import SelfSNN, SelfSNNConfig
from self_snn.utils.viz import save_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="runs/agency_s1")
    args = parser.parse_args()

    model = SelfSNN(SelfSNNConfig())
    out = model(steps=args.steps)

    router_stats = out["router_stats"]
    moe_ratio = float(out["moe_energy_ratio"])
    topk = router_stats["topk"]

    dense_synops_ratio = 1.0
    saving_pct = (1.0 - moe_ratio) * 100.0

    print("Router Top-K indices:", topk.tolist())
    print(f"MoE 条件计算能耗比 (masked/dense synops): {moe_ratio:.3f}")
    print(f"相比密集前向 synops 下降约 {saving_pct:.1f}%")

    # 画一个简单的能耗对比曲线：dense vs masked
    figs_dir = Path(args.logdir) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    save_curve(
        [dense_synops_ratio, moe_ratio],
        figs_dir / "energy_curve.png",
        xlabel="mode (0:dense,1:masked)",
        ylabel="relative_synops",
        title="Dense vs Masked Synops Ratio",
    )


if __name__ == "__main__":
    main()
