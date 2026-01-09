#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot AUROC distribution from a CSV-formatted file.")
    parser.add_argument("input", type=str, help="Path to a CSV-formatted file (e.g. .csv or .civ)")
    parser.add_argument("--out", type=str, default="", help="Output image path (png). Default: <input>_auroc_dist.png")
    parser.add_argument("--bins", type=int, default=20, help="Number of histogram bins")
    parser.add_argument("--title", type=str, default="AUROC distribution", help="Plot title")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_auroc_dist.png")

    # Read as CSV (assuming the file has a header row)
    df = pd.read_csv(in_path)

    if "auroc" not in df.columns:
        raise ValueError(f"'auroc' column not found. Available columns: {list(df.columns)}")

    auroc = pd.to_numeric(df["auroc"], errors="coerce").dropna().to_numpy()
    if auroc.size == 0:
        raise ValueError("No valid numeric values in 'auroc' column.")

    # Basic stats
    stats = {
        "count": auroc.size,
        "min": float(np.min(auroc)),
        "max": float(np.max(auroc)),
        "mean": float(np.mean(auroc)),
        "median": float(np.median(auroc)),
        "std": float(np.std(auroc, ddof=1)) if auroc.size >= 2 else float("nan"),
    }
    print("AUROC stats:", stats)

    # Plot: histogram + boxplot (2 rows)
    fig = plt.figure(figsize=(8, 6))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.hist(auroc, bins=args.bins, edgecolor="black")
    ax1.set_title(args.title)
    ax1.set_xlabel("AUROC")
    ax1.set_ylabel("Count")
    ax1.set_xlim(0.0, 1.0)

    # Add vertical lines for mean/median
    ax1.axvline(stats["mean"], linewidth=1.5, linestyle="--", label=f"mean={stats['mean']:.3f}")
    ax1.axvline(stats["median"], linewidth=1.5, linestyle=":", label=f"median={stats['median']:.3f}")
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.boxplot(auroc, vert=False, showmeans=True)
    ax2.set_xlabel("AUROC")
    ax2.set_yticks([1])
    ax2.set_yticklabels(["AUROC"])
    ax2.set_xlim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")

    # If you want to show interactively:
    # plt.show()


if __name__ == "__main__":
    main()
