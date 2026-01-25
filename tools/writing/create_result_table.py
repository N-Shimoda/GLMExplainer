import argparse
import statistics

import wandb

SUBSET_LABELS = {
    "ba_shapes": "BA-Shapes",
    "tree_cycle": "Tree-Cycle",
    "tree_grid": "Tree-Grid",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", choices=["auroc", "jaccard"], default="auroc")
    parser.add_argument("--precision", type=int, default=3)
    args = parser.parse_args()

    # Fetch runs from Weights & Biases
    api = wandb.Api()
    runs = api.runs(
        "naos-ku/MotifQA-Explainer",
        filters={
            "tags": {"$all": ["master", "full"]},
        },
    )
    baselines = [run for run in runs if run.config.get("llr_threshold") == 0]
    ours = [run for run in runs if run.config.get("llr_threshold", 0) > 0]

    # Metric configuration
    match args.metrics:
        case "auroc":
            metric_keys = ["avg_auroc"]
        case "jaccard":
            metric_keys = ["edge_mask_jaccard"]

    header_cells = ["Subset", "Ours", "Baseline"]
    rows: list[list[str]] = []

    # Prepare table data
    baseline_metrics = _collect_by_subset(baselines, metric_keys)
    ours_metrics = _collect_by_subset(ours, metric_keys)

    for subset, label in SUBSET_LABELS.items():
        ours_row = ours_metrics.get(subset, {})
        baseline_row = baseline_metrics.get(subset, {})
        ours_value = ours_row.get(metric_keys[0])
        baseline_value = baseline_row.get(metric_keys[0])
        rows.append(
            [
                label,
                _format(ours_value, args.precision),
                _format(baseline_value, args.precision),
            ]
        )

    # Calculate column widths
    col_widths = [
        max(len(cell), max((len(row[idx]) for row in rows), default=0)) for idx, cell in enumerate(header_cells)
    ]

    def _format_row(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(col_widths[idx]) for idx, cell in enumerate(cells)) + " |"

    separator = "| " + " | ".join("-" * width for width in col_widths) + " |"

    # Print the markdown table
    print(f"\n## Comparison of {args.metrics.upper()}\n")
    print("```markdown")
    print(_format_row(header_cells))
    print(separator)
    for row in rows:
        print(_format_row(row))
    print("```")


if __name__ == "__main__":
    main()
