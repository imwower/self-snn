import argparse
from pathlib import Path

import numpy as np


def gen_words_images(outdir: Path, n: int = 32) -> None:
    words = np.array([f"w{i}" for i in range(n)], dtype=object)
    images = np.random.randn(n, 1, 16, 16).astype("float32")
    np.savez(outdir / "words_images.npz", words=words, images=images)


def gen_sentences_images(outdir: Path, n: int = 32) -> None:
    sentences = np.array([f"句子 {i}" for i in range(n)], dtype=object)
    images = np.random.randn(n, 1, 16, 16).astype("float32")
    np.savez(outdir / "sentences_images.npz", sentences=sentences, images=images)


def gen_video_events(outdir: Path, n: int = 16, T: int = 20) -> None:
    videos = np.random.randn(n, T, 1, 16, 16).astype("float32")
    labels = np.arange(n, dtype="int64")
    np.savez(outdir / "video_events.npz", videos=videos, labels=labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="data/toy")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.task == "words_images":
        gen_words_images(outdir)
    elif args.task == "sentences_images":
        gen_sentences_images(outdir)
    elif args.task == "video_events":
        gen_video_events(outdir)
    else:
        raise ValueError(f"unknown task {args.task}")

    print(f"generated toy data for {args.task} at {outdir}")


if __name__ == "__main__":
    main()
