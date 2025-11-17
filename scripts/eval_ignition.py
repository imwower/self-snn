import argparse
from pathlib import Path

from self_snn.core.workspace import SelfSNN, SelfSNNConfig
from self_snn.utils.viz import save_raster, save_curve, save_heatmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs/s0_minimal")
    args = parser.parse_args()

    model = SelfSNN(SelfSNNConfig())
    out = model(steps=300)
    spikes = out["spikes"]
    vm_trace = out.get("vm_trace", None)
    ignite_mask = out.get("ignite_mask", None)
    rate = float(spikes.float().mean() * 1000.0)
    ignition_rate = float(out["ignition_rate"])
    kappa = float(out["branching_kappa"])

    figs_dir = Path(args.logdir) / "figs"
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

    print(
        f"Ignition eval: mean_rate={rate:.3f} Hz, "
        f"ignition_rate={ignition_rate:.3f}, branching_kappa={kappa:.3f}, "
        f"figs saved to {figs_dir}"
    )


if __name__ == "__main__":
    main()
