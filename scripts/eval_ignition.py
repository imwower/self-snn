import argparse
from pathlib import Path

from self_snn.core.workspace import SelfSNN, SelfSNNConfig
from self_snn.utils.viz import save_raster


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs/s0_minimal")
    args = parser.parse_args()

    model = SelfSNN(SelfSNNConfig())
    out = model(steps=300)
    spikes = out["spikes"]
    rate = float(spikes.float().mean() * 1000.0)
    ignition_rate = float(out["ignition_rate"])
    kappa = float(out["branching_kappa"])

    figs_dir = Path(args.logdir) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    save_raster(spikes, figs_dir / "raster_spikes.png")

    print(
        f"Ignition eval: mean_rate={rate:.3f} Hz, "
        f"ignition_rate={ignition_rate:.3f}, branching_kappa={kappa:.3f}, "
        f"raster saved to {figs_dir/'raster_spikes.png'}"
    )


if __name__ == "__main__":
    main()
