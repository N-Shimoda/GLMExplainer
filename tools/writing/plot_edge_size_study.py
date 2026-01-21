import argparse
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import wandb


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-format", type=str, default="pdf", choices=["pdf", "svg"])
    return p.parse_args()


def get_runs():
    api = wandb.Api()
    runs = api.runs(
        "naos-ku/MotifQA-Explainer",
        filters={
            "tags": {"$in": ["master"]},
        },
    )
    baselines = [run for run in runs if run.config.get("llr_threshold") == 0]
    ours = [run for run in runs if run.config.get("llr_threshold", 0) > 0]

    return baselines, ours


def plot_figure(
    df: pd.DataFrame,
    x_col: str = "edge size",
    y1_col: str = "AUROC",
    y2_col: str = "Jaccard",
    sort_by_x: bool = True,
    format: Literal["pdf", "svg"] = "pdf",
):
    # 必要な列チェック
    for col in (x_col, y1_col, y2_col):
        if col not in df.columns:
            raise ValueError(f"'{col}' 列が見つかりません。現状の列: {list(df.columns)}")

    # 数値化（文字列でもOKにする）
    d = df[[x_col, y1_col, y2_col]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y1_col] = pd.to_numeric(d[y1_col], errors="coerce")
    d[y2_col] = pd.to_numeric(d[y2_col], errors="coerce")
    d = d.dropna(subset=[x_col, y1_col, y2_col])

    if sort_by_x:
        d = d.sort_values(by=x_col)

    plt.figure()
    plt.plot(d[x_col], d[y1_col], label="AUROC", color="blue")
    plt.plot(d[x_col], d[y2_col], label="Jaccard", color="orange")

    plt.xlabel("edge size")
    plt.ylabel("AUROC")
    plt.title("Edge Size vs AUROC")

    plt.tight_layout()
    plt.savefig("edge_size_vs_auroc.pdf")


def main():
    args = build_args()

    # baselines, ours = get_runs()

    # for run in baselines + ours:
    #     print("Edge size:", run.config.get("edge_size"))

    df = pd.DataFrame(
        {
            "edge size": [10, 20, 30, 40, 50],
            "AUROC": [0.62, 0.70, 0.73, 0.75, 0.78],
            "Jaccard": [0.15, 0.18, 0.21, 0.20, 0.23],
        }
    )
    plot_figure(df, format=args.output_format)


if __name__ == "__main__":
    main()
