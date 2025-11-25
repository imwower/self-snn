import argparse
import json
from pathlib import Path

from self_snn.core.workspace import SelfSNN, SelfSNNConfig
from self_snn.utils.viz import save_curve
from self_snn.utils.logging_cn import setup_logger, log_energy, log_router
from self_snn.utils.config_loader import load_config, build_self_snn_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="YAML 配置路径，留空则使用默认 SelfSNNConfig")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="runs/agency_s1")
    parser.add_argument("--json", action="store_true", help="输出关键指标的 JSON")
    args = parser.parse_args()

    cfg = None
    model_cfg = SelfSNNConfig()
    if args.config:
        cfg = load_config(args.config)
        model_cfg = build_self_snn_config(cfg)

    logdir = args.logdir
    if not logdir and cfg is not None:
        logdir = cfg.get("logging", {}).get("logdir", "runs/agency_s1")

    logger = setup_logger(logdir)

    model = SelfSNN(model_cfg)
    out = model(steps=args.steps)

    router_stats = out["router_stats"]
    moe_ratio = float(out["moe_energy_ratio"])
    topk = router_stats["topk"]

    dense_synops_ratio = 1.0
    saving_pct = (1.0 - moe_ratio) * 100.0

    spikes = out.get("spikes")
    dt_ms = float(model.config.dt_ms if hasattr(model, "config") else 1.0)
    avg_spikes_per_s = 0.0
    if spikes is not None:
        avg_spikes_per_s = float(spikes.float().mean() * 1000.0 / max(dt_ms, 1e-6))

    energy_stats = out.get("energy_stats", None)
    synops_dense = float(getattr(energy_stats, "synops_dense", 0.0) if energy_stats is not None else 0.0)
    synops_sparse = float(getattr(energy_stats, "synops_masked", 0.0) if energy_stats is not None else 0.0)

    metrics = {
        "synops_dense": synops_dense,
        "synops_sparse": synops_sparse,
        "synops_ratio": moe_ratio,
        "avg_spikes_per_s": avg_spikes_per_s,
        "topk": topk.tolist(),
    }

    if not args.json:
        print("Router Top-K indices:", topk.tolist())
        print(f"MoE 条件计算能耗比 (masked/dense synops): {moe_ratio:.3f}")
        print(f"相比密集前向 synops 下降约 {saving_pct:.1f}%")
    else:
        print(json.dumps(metrics, ensure_ascii=False))

    # 中文日志与节能对照报告
    log_router(logger, f"Top-K 专家={topk.tolist()}；MoE 条件能耗比={moe_ratio:.3f}")
    log_energy(
        logger,
        f"条件计算能耗比={moe_ratio:.3f}（相对密集↓ {saving_pct:.1f}%；目标≥60%）",
    )

    # 文本报告，便于快速对比节能达标情况
    report_path = Path(logdir) / "router_energy_report.txt"
    target_saving = 60.0
    passed = saving_pct >= target_saving
    with report_path.open("w", encoding="utf-8") as f:
        f.write("【路由能耗对照报告】\n")
        f.write(f"Top-K 专家索引: {topk.tolist()}\n")
        f.write(f"MoE 条件能耗比 (masked/dense synops): {moe_ratio:.3f}\n")
        f.write(f"synops 相比密集前向下降: {saving_pct:.1f}%\n")
        f.write(f"是否达到节能目标(≥{target_saving:.1f}%): {'是' if passed else '否'}\n")

    # 画一个简单的能耗对比曲线：dense vs masked
    figs_dir = Path(logdir) / "figs"
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
