import argparse
import json
from pathlib import Path

import torch

from self_snn.core.workspace import SelfSNN, SelfSNNConfig
from self_snn.utils.viz import save_curve
from self_snn.utils.logging_cn import setup_logger, log_memory_write, log_memory_read
from self_snn.meta.plasticity import STDPConfig
from self_snn.utils.config_loader import load_config, build_self_snn_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="YAML 配置路径，留空则使用默认 SelfSNNConfig")
    parser.add_argument("--logdir", type=str, default="runs/agency_s1")
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--delay", type=int, default=5)
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
    key = model.self_model.key
    stdp_cfg = STDPConfig()

    hit_flags = []
    timing_errors = []

    for _ in range(args.trials):
        seq = torch.randn(args.seq_len, 8)
        model.memory.write(key, seq, delay=args.delay)
        log_memory_write(logger, f"写入序列，长度={args.seq_len}，标注延迟={args.delay}")
        recalled = model.memory.read(key)
        replayed, err_ms = model.memory.read_with_timing_error(key, max_len=args.seq_len + args.delay)
        hit = recalled is not None and replayed is not None
        hit_flags.append(1.0 if hit else 0.0)
        timing_errors.append(err_ms)

        # 简单调用 STDP 接口：使用序列首/末状态作为 pre/post 占位，并用时间误差的反比作为第三因子
        pre = seq[0]
        post = seq[-1]
        # 第三因子：误差越小，强化越强
        third = 1.0 / (1.0 + err_ms)
        model.memory.update_weights(key, pre, post, stdp_cfg, third_factor=third)

    hit_rate = sum(hit_flags) / max(len(hit_flags), 1)
    mean_err = sum(timing_errors) / max(len(timing_errors), 1)

    figs_dir = Path(logdir) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    # 使用 timing_errors 曲线近似 delay_replay_error，可视化 RMS 级误差
    save_curve(
        timing_errors,
        figs_dir / "delay_replay_error.png",
        xlabel="trial",
        ylabel="error_ms",
        title="Delay Replay Error",
    )

    mem_stats = model.memory.stats()

    log_memory_read(
        logger,
        f"命中率={hit_rate:.3f}, 平均重放误差={mean_err:.3f} ms, "
        f"键数={mem_stats['n_keys']:.0f}, 平均延迟={mem_stats['mean_delay']:.1f}, 方差={mem_stats['var_delay']:.1f}",
    )

    metrics = {
        "cue_hit_rate": float(hit_rate),
        "replay_time_error_ms": float(mean_err),
        "mean_delay_steps": float(mem_stats.get("mean_delay", 0.0)),
        "var_delay": float(mem_stats.get("var_delay", 0.0)),
    }

    if args.json:
        print(json.dumps(metrics, ensure_ascii=False))
    else:
        print(
            f"Memory eval: 命中率={hit_rate:.3f}, "
            f"平均重放时间误差={mean_err:.3f} ms, "
            f"平均延迟={mem_stats['mean_delay']:.1f} 步, "
            f"图已保存到 {figs_dir / 'delay_replay_error.png'}"
        )


if __name__ == "__main__":
    main()
