import argparse
import json
from pathlib import Path

from self_snn.core.workspace import SelfSNN, SelfSNNConfig
from self_snn.utils.viz import save_raster, save_curve, save_heatmap
from self_snn.utils.config_loader import load_config, build_self_snn_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="YAML 配置路径，留空则使用默认 SelfSNNConfig")
    parser.add_argument("--logdir", type=str, default="runs/s0_minimal")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--json", action="store_true", help="输出关键指标的 JSON")
    args = parser.parse_args()

    cfg = None
    model_cfg = SelfSNNConfig()
    if args.config:
        cfg = load_config(args.config)
        model_cfg = build_self_snn_config(cfg)

    logdir = args.logdir
    if not logdir and cfg is not None:
        logdir = cfg.get("logging", {}).get("logdir", "runs/s0_minimal")

    model = SelfSNN(model_cfg)
    out = model(steps=int(args.steps))
    spikes = out["spikes"]
    vm_trace = out.get("vm_trace", None)
    ignite_mask = out.get("ignite_mask", None)
    rate = float(spikes.float().mean() * 1000.0)
    ignition_rate = float(out["ignition_rate"])
    kappa = float(out["branching_kappa"])
    ignite_cov = out.get("ignite_coverage")
    ignite_cov_mean = float(ignite_cov.mean()) if ignite_cov is not None else 0.0

    figs_dir = Path(logdir) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    # 栅格图
    save_raster(spikes, figs_dir / "raster_spikes.png")
    # 膜电位平均轨迹
    if vm_trace is not None:
        mean_vm = vm_trace.float().mean(dim=1)
        save_curve(mean_vm, figs_dir / "vm_trace.png", xlabel="t", ylabel="vm", title="Vm Trace")
    # 点火热力图（时间×神经元）
    if ignite_mask is not None:
        save_heatmap(ignite_mask.float(), figs_dir / "ignition_heatmap.png", xlabel="neuron", ylabel="t", title="Ignition Heatmap")

    metrics = {
        "ignition_rate": ignition_rate,
        "branching_kappa": kappa,
        "mean_spike_rate_hz": rate,
        "ignite_coverage": ignite_cov_mean,
    }

    if args.json:
        print(json.dumps(metrics, ensure_ascii=False))
    else:
        print(
            f"Ignition eval: mean_rate={rate:.3f} Hz, "
            f"ignition_rate={ignition_rate:.3f}, coverage={ignite_cov_mean:.3f}, branching_kappa={kappa:.3f}, "
            f"figs saved to {figs_dir}"
        )


if __name__ == "__main__":
    main()
