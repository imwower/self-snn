import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="data/toy")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.task == "words_images":
        np.save(outdir / "words.npy", np.arange(10))
    print(f"generated toy data for {args.task} at {outdir}")


if __name__ == "__main__":
    main()

