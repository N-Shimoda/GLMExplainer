import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterable

import torch
from .metrics import (
    EDGE_MASK_STABILITY_KEYS,
    compute_edge_mask_stability_metrics_per_sample,
)


def _write_metrics_header(log_path: str, fieldnames: list[str]) -> None:
    with open(log_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def write_average_metrics_csv(
    output_path: str | Path,
    sample_metrics: Mapping[int, Mapping[str, float]] | None,
    edge_masks: Mapping[int, Sequence[Any]] | None,
) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
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

    Returns
    -------
    per_sample_stability : dict[int, dict[str, float]]
        Computed per-sample edge mask stability metrics.
    aggregated_stability : dict[str, float]
        Aggregated edge mask stability metrics averaged across samples.
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
            auroc_sum = stats.get("auroc_sum", 0.0)
            auprc_sum = stats.get("auprc_sum", 0.0)
            f1_sum = stats.get("f1_sum", 0.0)
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

    aggregated_stability = {key: 0.0 for key in EDGE_MASK_STABILITY_KEYS}
    if per_sample_stability:
        for key in EDGE_MASK_STABILITY_KEYS:
            aggregated_stability[key] = _mean([metrics.get(key, 0.0) for metrics in per_sample_stability.values()])

    return dict(per_sample_stability), aggregated_stability


def _compute_sample_average_row(
    sample_idx: int,
    stats: dict[str, float] | None,
    edge_masks: Iterable[torch.Tensor] | None,
) -> dict[str, float] | None:
    """Compute averaged accuracy and stability metrics for a single sample."""
    if not stats:
        return None
    count = int(stats.get("count", 0))
    if count <= 0:
        return None
    row: dict[str, float] = {
        "sample_index": sample_idx,
        "answer_accuracy": stats.get("answer_accuracy_sum", 0.0) / count,
        "auroc": stats.get("auroc_sum", 0.0) / count,
        "auprc": stats.get("auprc_sum", 0.0) / count,
        "f1": stats.get("f1_sum", 0.0) / count,
    }
    mask_list = list(edge_masks) if edge_masks is not None else []
    stability = compute_edge_mask_stability_metrics_per_sample({sample_idx: mask_list}).get(sample_idx, {})
    for key in EDGE_MASK_STABILITY_KEYS:
        row[key] = stability.get(key, 0.0)
    return row


def _record_sample_average_metrics(
    avg_log_path: str | None,
    fieldnames: list[str] | None,
    sample_idx: int,
    stats: dict[str, float] | None,
    edge_masks: Iterable[torch.Tensor] | None,
) -> None:
    """Append a per-sample averaged metrics row to the CSV log if possible."""
    if avg_log_path is None or fieldnames is None:
        return
    row = _compute_sample_average_row(sample_idx, stats, edge_masks)
    if row is None:
        return
    with open(avg_log_path, "a", newline="") as avg_file:
        writer = csv.DictWriter(avg_file, fieldnames=fieldnames)
        writer.writerow(row)


def append_run_history_row(history_path: str | Path, row: Mapping[str, Any]) -> None:
    """Append a run-level record to the shared history CSV.

    Parameters
    ----------
    history_path : str or pathlib.Path
        Destination CSV path (shared across runs for a subset).
    row : Mapping[str, Any]
        Dictionary containing the run metadata and metrics to record.
    """

    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "epochs",
        "lr",
        "edge_size",
        "edge_ent",
        "llr_threshold",
        "avg_auroc",
        "avg_auprc",
        "avg_f1",
        *EDGE_MASK_STABILITY_KEYS,
        "avg_answer_accuracy",
    ]

    is_new_file = not history_path.exists()
    with history_path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if is_new_file:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


__all__ = [
    "_write_metrics_header",
    "write_average_metrics_csv",
    "append_run_history_row",
    "_compute_sample_average_row",
    "_record_sample_average_metrics",
]
