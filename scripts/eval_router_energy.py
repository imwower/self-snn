import argparse
from pathlib import Path

from self_snn.core.workspace import SelfSNN, SelfSNNConfig
from self_snn.utils.viz import save_curve
from self_snn.utils.logging_cn import setup_logger, log_energy, log_router


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="runs/agency_s1")
    args = parser.parse_args()

    logger = setup_logger(args.logdir)

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

    # 中文日志与节能对照报告
    log_router(logger, f"Top-K 专家={topk.tolist()}；MoE 条件能耗比={moe_ratio:.3f}")
    log_energy(
        logger,
        f"条件计算能耗比={moe_ratio:.3f}（相对密集↓ {saving_pct:.1f}%；目标≥60%）",
    )

    # 文本报告，便于快速对比节能达标情况
    report_path = Path(args.logdir) / "router_energy_report.txt"
    target_saving = 60.0
    passed = saving_pct >= target_saving
    with report_path.open("w", encoding="utf-8") as f:
        f.write("【路由能耗对照报告】\n")
        f.write(f"Top-K 专家索引: {topk.tolist()}\n")
        f.write(f"MoE 条件能耗比 (masked/dense synops): {moe_ratio:.3f}\n")
        f.write(f"synops 相比密集前向下降: {saving_pct:.1f}%\n")
        f.write(f"是否达到节能目标(≥{target_saving:.1f}%): {'是' if passed else '否'}\n")

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
