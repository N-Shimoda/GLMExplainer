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
from tools.writing.wandb_cache import load_cached_runs, save_cached_runs  # noqa: E402


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", type=str, nargs="+", default=["edge_size", "small"], help="Wandb run tags to filter.")
    p.add_argument(
        "--use-cache",
        action="store_true",
        help="Load cached wandb runs from output dir instead of fetching from remote.",
    )
    p.add_argument(
        "--set-y-lim", action="store_true", help="Set y-axis limits based on min/max values across all runs."
    )
    p.add_argument("--output-dir", type=str, default="plots/edge_size_study")
    p.add_argument("--output-format", type=str, default="pdf", choices=["pdf", "svg"])

    # Debugging
    p.add_argument(
        "--test",
        action="store_true",
        help="Run a synthetic test of plot_figure() and save directly under plots/edge_size_study.",
    )
    return p.parse_args()


def run_test(args: argparse.Namespace):
    """Run a synthetic test of plot_figure()"""
    os.makedirs(args.output_dir, exist_ok=True)
    synthetic_complete = pd.DataFrame(
        {
            "edge size": [1, 2, 4, 8, 16],
            "AUROC": [0.61, 0.65, 0.7, 0.73, 0.76],
            "Jaccard": [0.2, 0.24, 0.27, 0.29, 0.31],
            "Spearman": [0.3, 0.35, 0.4, 0.43, 0.47],
        }
    )
    synthetic_empty = pd.DataFrame(
        {
            "edge size": [1, 2, 4, 8, 16],
            "AUROC": [0.6, 0.63, 0.67, 0.7, 0.73],
            "Jaccard": [0.19, 0.22, 0.25, 0.27, 0.29],
            "Spearman": [0.28, 0.32, 0.36, 0.39, 0.42],
        }
    )
    synthetic_baseline = pd.DataFrame(
        {
            "edge size": [1, 2, 4, 8, 16],
            "AUROC": [0.55, 0.58, 0.61, 0.63, 0.64],
            "Jaccard": [0.16, 0.18, 0.2, 0.21, 0.22],
            "Spearman": [0.22, 0.25, 0.28, 0.3, 0.32],
        }
    )
    filename = os.path.join(args.output_dir, f"auroc_test.{args.output_format}")
    plot_figure(
        synthetic_complete,
        synthetic_empty,
        synthetic_baseline,
        y_lim=None,
        metric="auroc",
        filename=filename,
    )
    print(f"Test plot saved to {filename}")


def _filter_runs_by_tags(runs: list[wandb.apis.public.Run], tags: list[str]) -> list[wandb.apis.public.Run]:
    if not tags:
        return runs
    tag_set = set(tags)
    return [run for run in runs if tag_set.issubset(set(run.tags or []))]


def get_wandb_runs(tags: list[str]):
    """Load wandb runs from MotifQA-Explainer project."""
    api = wandb.Api()
    filters = {
        "state": "finished",
        "config.edge_size": {"$exists": True},
    }
    if tags:
        filters["tags"] = {"$all": tags}
    runs = api.runs(
        "naos-ku/MotifQA-Explainer",
        filters=filters,
    )
    baselines = [run for run in runs if run.config.get("llr_threshold") == 0]
    ours = [run for run in runs if run.config.get("llr_threshold", 0) > 0]
    ours_complete = [run for run in ours if run.config.get("baseline_graph") == "complete"]
    ours_empty = [run for run in ours if run.config.get("baseline_graph") == "empty"]

    return baselines, ours, ours_complete, ours_empty


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
    df = pd.DataFrame({"edge size": [], "AUROC": [], "Jaccard": [], "Spearman": []})
    for run in subset_runs:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "edge size": [run.config["edge_size"]],
                        "AUROC": [run.summary["avg_auroc"]],
                        "Jaccard": [run.summary["edge_mask_jaccard"]],
                        "Spearman": [run.summary["edge_mask_spearman"]],
                    }
                ),
            ],
            ignore_index=True,
        )
    return df


def plot_figure(
    complete_df: pd.DataFrame,
    empty_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    y_lim: Optional[tuple[float, float]],
    metric: Literal["auroc", "jaccard", "spearman"],
    filename: str,
):
    """Plot baseline vs. complete/empty for a selected metric over edge sizes."""
    metric_col_map = {
        "auroc": "AUROC",
        "jaccard": "Jaccard",
        "spearman": "Spearman",
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

    complete_clean = _prepare(complete_df)
    empty_clean = _prepare(empty_df)
    baseline_clean = _prepare(baseline_df)

    plt.figure()
    plt.plot(
        complete_clean[x_col],
        complete_clean[y_col],
        label="complete",
        color="C1" if metric == "auroc" else "C0",
        linestyle="-",
        marker="o",
    )
    plt.plot(
        empty_clean[x_col],
        empty_clean[y_col],
        label="empty",
        color="C2" if metric == "auroc" else "C3",
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
    plt.xlabel(r"$\lambda_\mathrm{size}$", fontsize=16)
    plt.ylabel(y_col, fontsize=14)
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.ylim(y_lim)
    plt.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(filename)


def main():
    args = build_args()
    metric_mapping = {
        "auroc": "avg_auroc",
        "jaccard": "edge_mask_jaccard",
        "spearman": "edge_mask_spearman",
    }
    metrics = list(metric_mapping.keys())

    if args.test:
        run_test(args)
        return

    cache_path = os.path.join(args.output_dir, "wandb_runs_cache.json")

    # Create output directory
    for metric in metrics:
        dir_path = os.path.join(args.output_dir, metric)
        os.makedirs(dir_path, exist_ok=True)

    # Load runs from wandb
    if args.use_cache:
        baselines, ours_complete, ours_empty = load_cached_runs(cache_path)
        baselines = _filter_runs_by_tags(baselines, args.tags)
        ours_complete = _filter_runs_by_tags(ours_complete, args.tags)
        ours_empty = _filter_runs_by_tags(ours_empty, args.tags)
        print(f"Loaded cached runs from {cache_path}.")
    else:
        baselines, ours, ours_complete, ours_empty = get_wandb_runs(args.tags)
        print(
            "Loaded "
            f"{len(baselines)} baselines, {len(ours_complete)} complete, "
            f"and {len(ours_empty)} empty runs."
        )
        save_cached_runs(cache_path, baselines, ours_complete, ours_empty)
        print(f"Saved runs cache to {cache_path}.")

    # Report best runs
    for run_type, runs in [
        ("Baselines", baselines),
        ("Complete", ours_complete),
        ("Empty", ours_empty),
    ]:
        print(f"\n{run_type} best runs:")
        report_best_runs(runs, metric="avg_auroc")

    # Determine y-limits for each metric
    if args.set_y_lim:
        all_runs = baselines + ours_complete + ours_empty
        y_lim_dict = {
            metric: (
                min([run.summary[wandb_metric] for run in all_runs]),
                max([run.summary[wandb_metric] for run in all_runs]) + 0.005,
            )
            for metric, wandb_metric in metric_mapping.items()
        }
        print(y_lim_dict)
    else:
        y_lim_dict = None

    # Create plots
    for subset in MOTIFQA_SUBSETS:
        baseline_df = create_log_df(baselines, subset)
        complete_df = create_log_df(ours_complete, subset)
        empty_df = create_log_df(ours_empty, subset)
        for metric in metrics:
            filename = os.path.join(args.output_dir, metric, f"{subset}.{args.output_format}")
            plot_figure(
                complete_df,
                empty_df,
                baseline_df,
                y_lim=y_lim_dict[metric] if y_lim_dict else None,
                metric=metric,
                filename=filename,
            )
    print(f"\nPlots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
