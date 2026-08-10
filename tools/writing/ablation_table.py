import argparse
import statistics

import wandb

SUBSET_LABELS = {
    "ba_shapes": "BA-Shapes",
    "tree_cycle": "Tree-Cycle",
    "tree_grid": "Tree-Grid v1 (3x3)",
    "tree_grid_v2": "Tree-Grid v2 (2x3)",
    "ba_two_motifs": "BA-Two-Motifs",
    "shortest_path": "Shortest-Path",
}


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _format(value: float | None, precision: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def _collect_by_subset(runs: list[wandb.apis.public.Run], metrics: list[str]) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        subset = run.config.get("subset")
        if subset is None:
            continue
        summary = run.summary or {}
        if any(summary.get(metric_key) is None for metric_key in metrics):
            continue
        grouped.setdefault(subset, {metric_key: [] for metric_key in metrics})
        for metric_key in metrics:
            grouped[subset][metric_key].append(float(summary[metric_key]))

    return {
        subset: {metric_key: _mean(values[metric_key]) for metric_key in values} for subset, values in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print Markdown ablation tables (AUROC and Jaccard index) comparing runs "
            "with and without token selection, averaged over MotifQA subsets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tags", type=str, nargs="+", default=["fpai", "full"], help="Wandb run tags to filter (all must match)."
    )
    parser.add_argument("--precision", type=int, default=3, help="Number of decimal places for the metric values.")
    args = parser.parse_args()

    # Fetch runs from Weights & Biases
    api = wandb.Api()
    runs = api.runs(
        "naos-ku/MotifQA-Explainer",
        filters={"tags": {"$all": args.tags}},
    )
    baselines = [run for run in runs if run.config.get("llr_threshold") == 0]
    ours = [run for run in runs if run.config.get("llr_threshold", 0) > 0]

    header_cells = ["Subset", "w/ Token Selection", "w/o Token Selection"]

    def _build_rows(metric_key: str) -> list[list[str]]:
        rows: list[list[str]] = []
        baseline_metrics = _collect_by_subset(baselines, [metric_key])
        ours_metrics = _collect_by_subset(ours, [metric_key])
        for subset, label in SUBSET_LABELS.items():
            ours_row = ours_metrics.get(subset, {})
            baseline_row = baseline_metrics.get(subset, {})
            rows.append(
                [
                    label,
                    _format(ours_row.get(metric_key), args.precision),
                    _format(baseline_row.get(metric_key), args.precision),
                ]
            )
        return rows

    def _print_table(title: str, rows: list[list[str]]) -> None:
        col_widths = [
            max(len(cell), max((len(row[idx]) for row in rows), default=0)) for idx, cell in enumerate(header_cells)
        ]

        def _format_row(cells: list[str]) -> str:
            return "| " + " | ".join(cell.ljust(col_widths[idx]) for idx, cell in enumerate(cells)) + " |"

        separator = "| " + " | ".join("-" * width for width in col_widths) + " |"

        print(f"\n## {title}\n")
        print("```markdown")
        print(_format_row(header_cells))
        print(separator)
        for row in rows:
            print(_format_row(row))
        print("```")

    _print_table("Comparison of AUROC", _build_rows("avg_auroc"))
    _print_table("Comparison of Jaccard Index", _build_rows("edge_mask_jaccard"))


if __name__ == "__main__":
    main()
