import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .metrics import (
    EDGE_MASK_STABILITY_KEYS,
    compute_edge_mask_stability_metrics_per_sample,
)


def _write_metrics_header(log_path: str, fieldnames: list[str]) -> None:
    with open(log_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def write_average_metrics_csv(
    output_path: str | Path,
    sample_metrics: Mapping[int, Mapping[str, float]] | None,
    edge_masks: Mapping[int, Sequence[Any]] | None,
) -> None:
    """Write per-sample averaged metrics and stability scores to a CSV file.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Destination path for the CSV file.
    sample_metrics : Mapping[int, Mapping[str, float]] or None
        Aggregated per-sample statistic sums, keyed by sample index. Each value
        should provide at least ``count`` and metric sum entries (e.g.,
        ``answer_accuracy_sum``).
    edge_masks : Mapping[int, Sequence[Any]] or None
        Sequences of edge mask artifacts per sample index, used to derive
        stability metrics. Pass ``None`` to skip stability computation.
    """
    output_path = Path(output_path)
    fieldnames = [
        "sample_index",
        "answer_accuracy",
        "auroc",
        "auprc",
        "f1",
        *EDGE_MASK_STABILITY_KEYS,
    ]

    per_sample_stability = (
        compute_edge_mask_stability_metrics_per_sample(edge_masks or {}) if edge_masks is not None else {}
    )
    metrics_data = sample_metrics or {}
    zero_stability = {key: 0.0 for key in EDGE_MASK_STABILITY_KEYS}

    with output_path.open("w", newline="") as avg_file:
        writer = csv.DictWriter(avg_file, fieldnames=fieldnames)
        writer.writeheader()
        for sample_idx in sorted(metrics_data.keys()):
            stats = metrics_data[sample_idx]
            count = int(stats.get("count", 0))
            answer_acc_sum = stats.get("answer_accuracy_sum", 0.0)
            f1_sum = stats.get("f1_sum", 0.0)
            auroc_sum = stats.get("auroc_sum", 0.0)
            auprc_sum = stats.get("auprc_sum", 0.0)
            row = {
                "sample_index": sample_idx,
                "answer_accuracy": answer_acc_sum / count if count > 0 else 0.0,
                "auroc": auroc_sum / count if count > 0 else 0.0,
                "auprc": auprc_sum / count if count > 0 else 0.0,
                "f1": f1_sum / count if count > 0 else 0.0,
            }
            stability = per_sample_stability.get(sample_idx, zero_stability)
            for key in EDGE_MASK_STABILITY_KEYS:
                row[key] = stability.get(key, 0.0)
            writer.writerow(row)


__all__ = ["_write_metrics_header", "write_average_metrics_csv"]
