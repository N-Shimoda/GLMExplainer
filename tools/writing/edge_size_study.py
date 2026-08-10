import argparse
import math
import os
import sys
from datetime import UTC, datetime
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import wandb
from rich import box
from rich.console import Console
from rich.table import Table

console = Console()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.constants import MOTIFQA_SUBSETS  # noqa: E402
from tools.writing.wandb_cache import load_cached_runs, save_cached_runs  # noqa: E402


def build_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--tags", type=str, nargs="+", default=["fpai", "edge_size", "small"], help="Wandb run tags to filter."
    )
    p.add_argument(
        "--use-cache",
        action="store_true",
        help="Load cached wandb runs from output dir instead of fetching from remote.",
    )
    p.add_argument("--output-dir", type=str, default="plots/edge_size_study")
    p.add_argument("--output-format", type=str, default="svg", choices=["svg", "pdf"])

    p.add_argument(
        "--table-metric",
        type=str,
        default="auroc",
        choices=["auroc", "spearman"],
        help="Metric for the printed best-runs table (auroc or spearman).",
    )

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
            "Spearman": [0.3, 0.35, 0.4, 0.43, 0.47],
        }
    )
    synthetic_empty = pd.DataFrame(
        {
            "edge size": [1, 2, 4, 8, 16],
            "AUROC": [0.6, 0.63, 0.67, 0.7, 0.73],
            "Spearman": [0.28, 0.32, 0.36, 0.39, 0.42],
        }
    )
    synthetic_baseline = pd.DataFrame(
        {
            "edge size": [1, 2, 4, 8, 16],
            "AUROC": [0.55, 0.58, 0.61, 0.63, 0.64],
            "Spearman": [0.22, 0.25, 0.28, 0.3, 0.32],
        }
    )
    filename = os.path.join(args.output_dir, f"auroc_test.{args.output_format}")
    plot_figure(
        synthetic_complete,
        synthetic_empty,
        synthetic_baseline,
        metric="auroc",
        filename=filename,
    )
    print(f"Test plot saved to {filename}")


def _filter_runs_by_tags(runs: list[wandb.apis.public.Run], tags: list[str]) -> list[wandb.apis.public.Run]:
    if not tags:
        return runs
    tag_set = set(tags)
    return [run for run in runs if tag_set.issubset(set(getattr(run, "tags", []) or []))]


def get_wandb_runs(
    tags: list[str],
) -> tuple[list[wandb.apis.public.Run], list[wandb.apis.public.Run], list[wandb.apis.public.Run]]:
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

    def _get_llr_threshold(run) -> float:
        """Replace missing or invalid llr_threshold with 0.0 to classify baselines vs. ours."""
        value = run.config.get("llr_threshold")
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    baselines = [run for run in runs if _get_llr_threshold(run) == 0]
    ours = [run for run in runs if _get_llr_threshold(run) > 0]
    ours_complete = [run for run in ours if run.config.get("baseline_graph") == "complete"]
    ours_empty = [run for run in ours if run.config.get("baseline_graph") == "empty"]

    return baselines, ours_complete, ours_empty


def get_best_run(
    runs: list[wandb.apis.public.Run], metric: Literal["avg_auroc", "edge_mask_spearman"]
) -> wandb.apis.public.Run:
    """Get the run with the best value for the specified metric."""
    best_run = max(runs, key=lambda run: run.summary.get(metric, float("-inf")))
    return best_run


def report_best_runs(
    baselines,
    ours_complete,
    ours_empty,
    metric: Literal["avg_auroc", "edge_mask_spearman"] = "avg_auroc",
):
    def _format_metric(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.4f}"

    def _to_float(value) -> float | None:
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        # Treat NaN as missing
        if math.isnan(v):
            return None
        return v

    def _best_value_for_subset(runs, subset: str):
        subset_runs = [run for run in runs if run.config.get("subset") == subset]
        if not subset_runs:
            return (None, "n/a", "n/a")
        best_run = get_best_run(subset_runs, metric)
        run_name = best_run.name.split("_")[-1] if best_run.name else best_run.id
        best_val = _to_float(best_run.summary.get(metric))
        edge_size = best_run.config.get("edge_size")
        edge_size_str = str(edge_size) if edge_size is not None else "n/a"
        return (best_val, run_name, edge_size_str)

    rows = []
    for subset in MOTIFQA_SUBSETS:
        b_val, b_run, b_edge = _best_value_for_subset(baselines, subset)
        c_val, c_run, c_edge = _best_value_for_subset(ours_complete, subset)
        e_val, e_run, e_edge = _best_value_for_subset(ours_empty, subset)

        # If all are missing, skip the row entirely
        if b_val is None and c_val is None and e_val is None:
            continue

        rows.append(
            {
                "subset": subset,
                "baseline_val": b_val,
                "complete_val": c_val,
                "empty_val": e_val,
                "baseline_run": b_run,
                "complete_run": c_run,
                "empty_run": e_run,
                "baseline_edge": b_edge,
                "complete_edge": c_edge,
                "empty_edge": e_edge,
            }
        )

    if not rows:
        console.print("[yellow]No runs found for the specified subsets.[/yellow]")
        return

    table = Table(title=f"Best runs ({metric})", box=box.SIMPLE_HEAD)
    table.add_column("subset", style="bold")
    table.add_column("baseline", justify="right")
    table.add_column("complete", justify="right")
    table.add_column("empty", justify="right")

    def _highlight_if_max(num: float | None, run_name: str, edge_size: str, max_val: float | None) -> str:
        """Format the metric value and highlight if it's the max among the three."""
        if num is None:
            return "n/a"
        value_str = _format_metric(num)
        if max_val is not None and abs(num - max_val) < 1e-12:
            value_str = f"[bold green]{value_str}[/bold green]"
        return f"{value_str} ({run_name}, edge={edge_size})"

    for r in rows:
        nums = [r["baseline_val"], r["complete_val"], r["empty_val"]]
        present = [n for n in nums if n is not None]
        max_val = max(present) if present else None

        table.add_row(
            str(r["subset"]),
            _highlight_if_max(r["baseline_val"], r["baseline_run"], r["baseline_edge"], max_val),
            _highlight_if_max(r["complete_val"], r["complete_run"], r["complete_edge"], max_val),
            _highlight_if_max(r["empty_val"], r["empty_run"], r["empty_edge"], max_val),
        )

    console.print(table)


def create_log_df(runs, subset: str) -> pd.DataFrame:
    # Filter runs by subset
    subset_runs = [run for run in runs if run.config.get("subset") == subset]
    latest_runs_by_edge_size: dict[float | str, tuple[datetime, int, object]] = {}

    def _run_timestamp(run) -> datetime:
        # Prefer updated_at for "latest", then created_at; support cached runs lacking both.
        for attr in ["updated_at", "created_at"]:
            value = getattr(run, attr, None)
            if not value:
                continue
            try:
                parsed = datetime.fromisoformat(str(value))
            except ValueError:
                continue
            # Normalise to aware datetimes so naive and aware runs stay comparable.
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        return datetime.min.replace(tzinfo=UTC)

    def _edge_size_key(edge_size):
        try:
            return float(edge_size)
        except (TypeError, ValueError):
            return str(edge_size)

    for index, run in enumerate(subset_runs):
        edge_size_key = _edge_size_key(run.config.get("edge_size"))
        run_timestamp = _run_timestamp(run)
        prev = latest_runs_by_edge_size.get(edge_size_key)
        if prev is None or (run_timestamp, index) >= (prev[0], prev[1]):
            latest_runs_by_edge_size[edge_size_key] = (run_timestamp, index, run)

    # Create dataframe
    rows = []
    for _, _, run in latest_runs_by_edge_size.values():
        rows.append(
            {
                "edge size": run.config.get("edge_size"),
                "AUROC": run.summary.get("avg_auroc"),
                "Spearman": run.summary.get("edge_mask_spearman"),
            }
        )
    return pd.DataFrame(rows, columns=["edge size", "AUROC", "Spearman"])


def plot_figure(
    complete_df: pd.DataFrame,
    empty_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    metric: Literal["auroc", "spearman"],
    filename: str,
):
    """Plot baseline vs. complete/empty for a selected metric over edge sizes."""
    metric_col_map = {
        "auroc": "AUROC",
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
        color="C1",
        linestyle="-",
        marker="o",
    )
    plt.plot(
        empty_clean[x_col],
        empty_clean[y_col],
        label="empty",
        color="C2",
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
    plt.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(filename)


def main():
    args = build_args()
    metric_mapping = {
        "auroc": "avg_auroc",
        "spearman": "edge_mask_spearman",
    }
    metrics = list(metric_mapping.keys())

    print("Filtering wandb runs with tags:", args.tags)

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
        baselines, ours_complete, ours_empty = get_wandb_runs(args.tags)
        print(f"Loaded {len(baselines)} baselines, {len(ours_complete)} complete, and {len(ours_empty)} empty runs.")
        save_cached_runs(cache_path, baselines, ours_complete, ours_empty)
        print(f"Saved runs cache to {cache_path}.")

    # Report best runs (single table)
    print("\nBest runs:")
    table_metric = metric_mapping[args.table_metric]
    report_best_runs(baselines, ours_complete, ours_empty, metric=table_metric)

    # Create plots
    skipped = []
    for subset in MOTIFQA_SUBSETS:
        baseline_df = create_log_df(baselines, subset)
        complete_df = create_log_df(ours_complete, subset)
        empty_df = create_log_df(ours_empty, subset)
        if baseline_df.empty and complete_df.empty and empty_df.empty:
            skipped.append(subset)
            continue
        for metric in metrics:
            filename = os.path.join(args.output_dir, metric, f"{subset}.{args.output_format}")
            plot_figure(
                complete_df,
                empty_df,
                baseline_df,
                metric=metric,
                filename=filename,
            )
    print(f"\nPlots saved to {args.output_dir}")
    if skipped:
        print(f"Skipped subsets with no runs: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
