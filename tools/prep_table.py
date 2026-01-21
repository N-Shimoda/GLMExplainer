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
    args = parser.parse_args()

    metric_precision = 4

    match args.metrics:
        case "auroc":
            metric_keys = ["avg_auroc"]
            header = "| Subset | Ours: AUROC | Baseline: AUROC |"
            separator = "| --- | --- | --- |"
        case "jaccard":
            metric_keys = ["edge_mask_jaccard"]
            header = "| Subset | Ours: Jaccard | Baseline: Jaccard |"
            separator = "| --- | --- | --- |"

    rows = [header, separator]

    api = wandb.Api()
    runs = api.runs(
        "naos-ku/MotifQA-Explainer",
        filters={
            "config.edge_size": 10,
            "tags": {"$in": ["master"]},
        },
    )

    baselines = [run for run in runs if run.config.get("llr_threshold") == 0]
    ours = [run for run in runs if run.config.get("llr_threshold", 0) > 0]

    baseline_metrics = _collect_by_subset(baselines, metric_keys)
    ours_metrics = _collect_by_subset(ours, metric_keys)

    for subset, label in SUBSET_LABELS.items():
        ours_row = ours_metrics.get(subset, {})
        baseline_row = baseline_metrics.get(subset, {})
        ours_value = ours_row.get(metric_keys[0])
        baseline_value = baseline_row.get(metric_keys[0])
        rows.append(
            "| "
            + " | ".join(
                [
                    label,
                    _format(ours_value, metric_precision),
                    _format(baseline_value, metric_precision),
                ]
            )
            + " |"
        )

    print(f"## Comparison of {args.metrics.upper()} between Ours and Baseline\n")
    print("```markdown")
    print("\n".join(rows))
    print("```")


if __name__ == "__main__":
    main()
