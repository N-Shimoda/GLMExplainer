import argparse
import os
import sys
from typing import Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd
import wandb

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.constants import MOTIFQA_SUBSETS  # noqa: E402


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--set-y-lim", action="store_true", help="Set y-axis limits based on min/max values across all runs."
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="Run a synthetic test of plot_figure() and save directly under plots/edge_size_study.",
    )
    p.add_argument("--output-dir", type=str, default="plots/edge_size_study")
    p.add_argument("--output-format", type=str, default="pdf", choices=["pdf", "svg"])
    return p.parse_args()


def run_test(args: argparse.Namespace):
    """Run a synthetic test of plot_figure()"""
    os.makedirs(args.output_dir, exist_ok=True)
    synthetic_ours = pd.DataFrame(
        {
            "edge size": [1, 2, 4, 8, 16],
            "AUROC": [0.61, 0.65, 0.7, 0.73, 0.76],
            "Jaccard": [0.2, 0.24, 0.27, 0.29, 0.31],
        }
    )
    synthetic_baseline = pd.DataFrame(
        {
            "edge size": [1, 2, 4, 8, 16],
            "AUROC": [0.55, 0.58, 0.61, 0.63, 0.64],
            "Jaccard": [0.16, 0.18, 0.2, 0.21, 0.22],
        }
    )
    filename = os.path.join(args.output_dir, f"edge_size_study_test.{args.output_format}")
    plot_figure(
        synthetic_ours,
        synthetic_baseline,
        y_lim=None,
        metric="auroc",
        filename=filename,
    )
    print(f"Test plot saved to {filename}")


def get_runs():
    """Load wandb runs from MotifQA-Explainer project."""
    api = wandb.Api()
    runs = api.runs(
        "naos-ku/MotifQA-Explainer",
        filters={
            "tags": {"$all": ["master", "small"]},
            "state": "finished",
            "config.edge_size": {"$exists": True},
        },
    )
    baselines = [run for run in runs if run.config.get("llr_threshold") == 0]
    ours = [run for run in runs if run.config.get("llr_threshold", 0) > 0]

    return baselines, ours


def get_best_run(
    runs: list[wandb.apis.public.Run], metric: Literal["avg_auroc", "edge_mask_jaccard"]
) -> wandb.apis.public.Run:
    """Get the run with the best value for the specified metric."""
    best_run = max(runs, key=lambda run: run.summary.get(metric, float("-inf")))
    return best_run


def report_best_runs(runs, metric: Literal["avg_auroc", "edge_mask_jaccard"] = "avg_auroc"):
    def _format_metric(value):
        if value is None:
            return "n/a"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"{value:.4f}"
        return str(value)

    rows = []
    for subset in MOTIFQA_SUBSETS:
        subset_runs = [run for run in runs if run.config.get("subset") == subset]
        if not subset_runs:
            continue
        best_run = get_best_run(subset_runs, metric)
        rows.append(
            {
                "subset": subset,
                "metric": metric,
                "value": _format_metric(best_run.summary.get(metric)),
                "run": best_run.name or best_run.id,
                "edge_size": best_run.config.get("edge_size", "n/a"),
            }
        )

    if not rows:
        print("No runs found for the specified subsets.")
        return

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def create_log_df(runs, subset: str) -> pd.DataFrame:
    # Filter runs by subset
    subset_runs = [run for run in runs if run.config.get("subset") == subset]

    # Create dataframe
    df = pd.DataFrame({"edge size": [], "AUROC": [], "Jaccard": []})
    for run in subset_runs:
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


def plot_figure(
    ours_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    y_lim: Optional[tuple[float, float]],
    metric: Literal["auroc", "jaccard"],
    filename: str,
):
    """Plot baseline vs. ours for a selected metric over edge sizes."""
    metric_col_map = {
        "auroc": "AUROC",
        "jaccard": "Jaccard",
    }
    if metric not in metric_col_map:
        raise ValueError(f"Unsupported metric '{metric}'. Expected one of {list(metric_col_map)}")

    x_col = "edge size"
    y_col = metric_col_map[metric]

    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(
                f"Missing required columns in dataframe. Needed: '{x_col}', '{y_col}'. Available: {list(df.columns)}"
            )
        d = df[[x_col, y_col]].copy()
        d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
        d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
        return d.dropna(subset=[x_col, y_col]).sort_values(by=x_col)

    ours_clean = _prepare(ours_df)
    baseline_clean = _prepare(baseline_df)

    plt.figure()
    plt.plot(
        ours_clean[x_col],
        ours_clean[y_col],
        label="w/ token selection",
        color="C1" if metric == "auroc" else "C0",
        linestyle="-",
        marker="o",
    )
    plt.plot(
        baseline_clean[x_col],
        baseline_clean[y_col],
        label="w/o token selection",
        color="gray",
        linestyle="--",
        marker="o",
    )
    plt.grid(visible=True, which="both", linestyle="--", linewidth=0.5)

    plt.xscale("log")
    plt.xlabel(r"$\lambda_\mathrm{size}$")
    plt.ylabel(y_col)
    plt.ylim(y_lim)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)


def main():
    args = build_args()
    metric_mapping = {"auroc": "avg_auroc", "jaccard": "edge_mask_jaccard"}
    metrics = list(metric_mapping.keys())

    if args.test:
        run_test(args)
        return

    # Create output directory
    for metric in metrics:
        dir_path = os.path.join(args.output_dir, metric)
        os.makedirs(dir_path, exist_ok=True)

    # Load runs from wandb
    baselines, ours = get_runs()
    print(f"Loaded {len(baselines)} baselines and {len(ours)} ours runs.")

    # Report best runs
    for run_type, runs in [("Baselines", baselines), ("Ours", ours)]:
        print(f"\n{run_type} best runs:")
        report_best_runs(runs, metric="avg_auroc")

    # Determine y-limits for each metric
    if args.set_y_lim:
        y_lim_dict = {
            metric: (
                min([run.summary[wandb_metric] for run in baselines + ours]),
                max([run.summary[wandb_metric] for run in baselines + ours]) + 0.0005,
            )
            for metric, wandb_metric in metric_mapping.items()
        }
        print(y_lim_dict)
    else:
        y_lim_dict = None

    # Create plots
    for subset in MOTIFQA_SUBSETS:
        baseline_df = create_log_df(baselines, subset)
        ours_df = create_log_df(ours, subset)
        for metric in metrics:
            filename = os.path.join(args.output_dir, metric, f"{subset}.{args.output_format}")
            plot_figure(
                ours_df,
                baseline_df,
                y_lim=y_lim_dict[metric] if y_lim_dict else None,
                metric=metric,
                filename=filename,
            )
    print(f"\nPlots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
