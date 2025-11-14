import argparse
import sys
from pathlib import Path
from subprocess import run, CalledProcessError


def sh(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    try:
        run(cmd, check=True)
    except CalledProcessError as e:
        print(f"[WARN] command failed with code {e.returncode}: {' '.join(cmd)}")


def run_s0(base_logdir: Path, duration: int) -> None:
    logdir = base_logdir / "s0_minimal"
    sh(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            "configs/s0_minimal.yaml",
            "--logdir",
            str(logdir),
            "--duration",
            str(duration),
        ]
    )
    sh(
        [
            sys.executable,
            "scripts/eval_ignition.py",
            "--logdir",
            str(logdir),
        ]
    )


def run_s1(base_logdir: Path, duration: int, epochs_fig: int, steps_fig: int) -> None:
    logdir = base_logdir / "agency_s1"
    sh(
        [
            sys.executable,
            "scripts/gen_toy_data.py",
            "--task",
            "words_images",
        ]
    )
    sh(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            "configs/s1_words.yaml",
            "--logdir",
            str(logdir),
            "--duration",
            str(duration),
        ]
    )
    sh(
        [
            sys.executable,
            "scripts/eval_memory.py",
            "--logdir",
            str(logdir),
        ]
    )
    sh(
        [
            sys.executable,
            "scripts/eval_lang.py",
            "--logdir",
            str(logdir),
        ]
    )
    sh(
        [
            sys.executable,
            "scripts/eval_router_energy.py",
            "--steps",
            "100",
        ]
    )
    sh(
        [
            sys.executable,
            "scripts/eval_figs.py",
            "--logdir",
            str(logdir),
            "--epochs",
            str(epochs_fig),
            "--steps",
            str(steps_fig),
        ]
    )


def run_s2(base_logdir: Path, duration: int, epochs_fig: int, steps_fig: int) -> None:
    logdir = base_logdir / "agency_s2"
    sh(
        [
            sys.executable,
            "scripts/gen_toy_data.py",
            "--task",
            "sentences_images",
        ]
    )
    sh(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            "configs/s2_sentence.yaml",
            "--logdir",
            str(logdir),
            "--duration",
            str(duration),
        ]
    )
    sh(
        [
            sys.executable,
            "scripts/eval_sentence.py",
            "--logdir",
            str(logdir),
        ]
    )
    sh(
        [
            sys.executable,
            "scripts/eval_router_energy.py",
            "--steps",
            "100",
        ]
    )
    sh(
        [
            sys.executable,
            "scripts/eval_figs.py",
            "--logdir",
            str(logdir),
            "--epochs",
            str(epochs_fig),
            "--steps",
            str(steps_fig),
        ]
    )


def run_s3(base_logdir: Path, duration: int, epochs_fig: int, steps_fig: int) -> None:
    logdir = base_logdir / "agency_s3"
    sh(
        [
            sys.executable,
            "scripts/gen_toy_data.py",
            "--task",
            "video_events",
        ]
    )
    sh(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            "configs/s3_video.yaml",
            "--logdir",
            str(logdir),
            "--duration",
            str(duration),
        ]
    )
    sh(
        [
            sys.executable,
            "scripts/eval_video.py",
            "--logdir",
            str(logdir),
        ]
    )
    sh(
        [
            sys.executable,
            "scripts/eval_router_energy.py",
            "--steps",
            "100",
        ]
    )
    # 睡眠 / 巩固 + 主体性长期曲线
    sh(
        [
            sys.executable,
            "scripts/sleep_consolidate.py",
            "--logdir",
            str(logdir),
            "--epochs",
            "10",
            "--steps",
            "100",
        ]
    )
    sh(
        [
            sys.executable,
            "scripts/eval_figs.py",
            "--logdir",
            str(logdir),
            "--epochs",
            str(epochs_fig),
            "--steps",
            str(steps_fig),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["s0", "s1", "s2", "s3", "all"],
        help="选择要运行的场景：S0/S1/S2/S3 或 all",
    )
    parser.add_argument("--base_logdir", type=str, default="runs")
    parser.add_argument("--duration", type=int, default=180, help="每次 train 的时间步数（秒级可粗略等于 steps）")
    parser.add_argument("--epochs_fig", type=int, default=50, help="eval_figs 中用于画曲线的 epoch 数")
    parser.add_argument("--steps_fig", type=int, default=20, help="eval_figs 中每个 epoch 的步数")
    args = parser.parse_args()

    base_logdir = Path(args.base_logdir)

    if args.scenario in ("s0", "all"):
        run_s0(base_logdir, args.duration)
    if args.scenario in ("s1", "all"):
        run_s1(base_logdir, args.duration, args.epochs_fig, args.steps_fig)
    if args.scenario in ("s2", "all"):
        run_s2(base_logdir, args.duration, args.epochs_fig, args.steps_fig)
    if args.scenario in ("s3", "all"):
        run_s3(base_logdir, args.duration, args.epochs_fig, args.steps_fig)


if __name__ == "__main__":
    main()

