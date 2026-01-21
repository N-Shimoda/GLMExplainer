import argparse
import os
import sys
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import wandb

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.constants import MOTIFQA_SUBSETS  # noqa: E402


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default="plots/edge_size_study")
    p.add_argument("--output-format", type=str, default="pdf", choices=["pdf", "svg"])
    return p.parse_args()


def get_runs():
    """Load wandb runs from MotifQA-Explainer project."""
    api = wandb.Api()
    runs = api.runs(
        "naos-ku/MotifQA-Explainer",
        filters={
            "tags": {"$in": ["master"]},
            "state": "finished",
            "config.edge_size": {"$exists": True},
        },
    )
    baselines = [run for run in runs if run.config.get("llr_threshold") == 0]
    ours = [run for run in runs if run.config.get("llr_threshold", 0) > 0]

    print(ours[0].config)

    return baselines, ours


def create_log_df(runs):
    df = pd.DataFrame(
        {
            "edge size": [],
            "AUROC": [],
            "Jaccard": [],
        }
    )
    for run in runs:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "edge size": [run.config["edge_size"]],
                        "AUROC": [run.summary["avg_auroc"]],
                        "Jaccard": [run.summary["edge_mask_jaccard"]],
                    }
                ),
            ],
            ignore_index=True,
        )
    return df


def plot_figure(df: pd.DataFrame, run_type: Literal["baseline", "ours"], filename: str):
    x_col = "edge size"
    y1_col = "AUROC"
    y2_col = "Jaccard"

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

    d = d.sort_values(by=x_col)

    plt.figure()
    plt.plot(d[x_col], d[y1_col], label="AUROC", color="blue")
    plt.plot(d[x_col], d[y2_col], label="Jaccard Index", color="orange")

    plt.xscale("log")
    plt.xlabel(r"$\lambda_\mathrm{size}$")
    plt.ylabel("AUROC")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)


def main():
    args = build_args()

    os.makedirs(args.output_dir, exist_ok=True)

    baselines, ours = get_runs()
    baselines_dict = {
        subset: [run for run in baselines if run.config.get("subset") == subset] for subset in MOTIFQA_SUBSETS
    }
    ours_dict = {subset: [run for run in ours if run.config.get("subset") == subset] for subset in MOTIFQA_SUBSETS}
    print(f"Loaded {len(baselines)} baselines and {len(ours)} ours runs.")

    for subset in MOTIFQA_SUBSETS:
        print(f"Subset: {subset}")
        baseline_df = create_log_df(baselines_dict[subset])
        ours_df = create_log_df(ours_dict[subset])
        plot_figure(
            baseline_df,
            run_type="baseline",
            filename=os.path.join(args.output_dir, f"{subset}_baseline.{args.output_format}"),
        )
        plot_figure(
            ours_df,
            run_type="ours",
            filename=os.path.join(args.output_dir, f"{subset}_ours.{args.output_format}"),
        )


if __name__ == "__main__":
    main()
