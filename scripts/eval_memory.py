import argparse
from pathlib import Path

import torch

from self_snn.core.workspace import SelfSNN, SelfSNNConfig
from self_snn.utils.viz import save_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs/agency_s1")
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--delay", type=int, default=5)
    args = parser.parse_args()

    model = SelfSNN(SelfSNNConfig())
    key = model.self_model.key

    hit_flags = []
    timing_errors = []

    for _ in range(args.trials):
        seq = torch.randn(args.seq_len, 8)
        model.memory.write(key, seq, delay=args.delay)
        recalled = model.memory.read(key)
        replayed, err_ms = model.memory.read_with_timing_error(key, max_len=args.seq_len + args.delay)
        hit = recalled is not None and replayed is not None
        hit_flags.append(1.0 if hit else 0.0)
        timing_errors.append(err_ms)

    hit_rate = sum(hit_flags) / max(len(hit_flags), 1)
    mean_err = sum(timing_errors) / max(len(timing_errors), 1)

    figs_dir = Path(args.logdir) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    # 使用 timing_errors 曲线近似 delay_replay_error，可视化 RMS 级误差
    save_curve(timing_errors, figs_dir / "delay_replay_error.png", xlabel="trial", ylabel="error_ms", title="Delay Replay Error")

    print(
        f"Memory eval: 命中率={hit_rate:.3f}, "
        f"平均重放时间误差={mean_err:.3f} ms, "
        f"图已保存到 {figs_dir / 'delay_replay_error.png'}"
    )


if __name__ == "__main__":
    main()
