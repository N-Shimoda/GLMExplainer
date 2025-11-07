"""Utility to derive per-sample averages from explanation trial metrics.

This script reads the per-trial ``sample_metrics.csv`` produced by ``explain.py``
and writes the aggregated per-sample averages to ``average_metrics.csv`` in the
same directory (or a user-specified output path).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.explanation.logging import write_average_metrics_csv  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create average_metrics.csv from a sample_metrics.csv file.")
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the sample_metrics.csv file produced by explain.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination for the average_metrics.csv file. Defaults to the input directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing average_metrics.csv at the destination.",
    )
    return parser.parse_args()


def _load_sample_metrics(csv_path: Path) -> Dict[int, Dict[str, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    required_fields = {"sample_index", "answer_accuracy", "auroc", "auprc", "f1"}
    aggregates: Dict[int, Dict[str, float]] = defaultdict(
        lambda: {
            "answer_accuracy_sum": 0.0,
            "auroc_sum": 0.0,
            "auprc_sum": 0.0,
            "f1_sum": 0.0,
            "count": 0,
        }
    )

    with csv_path.open(newline="") as infile:
        reader = csv.DictReader(infile)
        missing = required_fields - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            sample_idx = int(row["sample_index"])
            stats = aggregates[sample_idx]
            stats["answer_accuracy_sum"] += float(row["answer_accuracy"])
            stats["auroc_sum"] += float(row["auroc"])
            stats["auprc_sum"] += float(row["auprc"])
            stats["f1_sum"] += float(row["f1"])
            stats["count"] += 1

    return {idx: dict(values) for idx, values in aggregates.items()}


def _compute_overall_averages(metrics: Dict[int, Dict[str, float]]) -> dict[str, float]:
    total_count = 0
    totals = {
        "answer_accuracy_sum": 0.0,
        "auroc_sum": 0.0,
        "auprc_sum": 0.0,
        "f1_sum": 0.0,
    }
    for stats in metrics.values():
        count = int(stats.get("count", 0))
        total_count += count
        totals["answer_accuracy_sum"] += stats.get("answer_accuracy_sum", 0.0)
        totals["auroc_sum"] += stats.get("auroc_sum", 0.0)
        totals["auprc_sum"] += stats.get("auprc_sum", 0.0)
        totals["f1_sum"] += stats.get("f1_sum", 0.0)

    averages = {
        "avg_answer_accuracy": 0.0,
        "avg_auroc": 0.0,
        "avg_auprc": 0.0,
        "avg_f1": 0.0,
        "total_count": total_count,
    }
    if total_count > 0:
        averages["avg_answer_accuracy"] = totals["answer_accuracy_sum"] / total_count
        averages["avg_auroc"] = totals["auroc_sum"] / total_count
        averages["avg_auprc"] = totals["auprc_sum"] / total_count
        averages["avg_f1"] = totals["f1_sum"] / total_count
    return averages


def _compute_samplewise_averages(metrics: Dict[int, Dict[str, float]]) -> dict[str, float]:
    sample_totals = {
        "answer_accuracy": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "f1": 0.0,
    }
    sample_count = 0
    for stats in metrics.values():
        count = int(stats.get("count", 0))
        if count <= 0:
            continue
        sample_count += 1
        sample_totals["answer_accuracy"] += stats.get("answer_accuracy_sum", 0.0) / count
        sample_totals["auroc"] += stats.get("auroc_sum", 0.0) / count
        sample_totals["auprc"] += stats.get("auprc_sum", 0.0) / count
        sample_totals["f1"] += stats.get("f1_sum", 0.0) / count

    averages = {
        "answer_accuracy": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "f1": 0.0,
        "sample_count": sample_count,
    }
    if sample_count > 0:
        averages["answer_accuracy"] = sample_totals["answer_accuracy"] / sample_count
        averages["auroc"] = sample_totals["auroc"] / sample_count
        averages["auprc"] = sample_totals["auprc"] / sample_count
        averages["f1"] = sample_totals["f1"] / sample_count
    return averages


def main() -> None:
    args = _parse_args()
    input_csv = args.input_csv.resolve()
    output_target = args.output or input_csv.with_name("average_metrics.csv")
    output_csv = output_target.resolve()
    if output_csv.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file {output_csv}. Re-run with --overwrite to replace it."
        )
    metrics = _load_sample_metrics(input_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _, aggregated_stability = write_average_metrics_csv(output_csv, metrics, edge_masks=None)
    averages = _compute_overall_averages(metrics)
    total_count = averages.pop("total_count")
    if total_count > 0:
        print(
            "Average explanation accuracy across positive samples: "
            f"AnswerAcc={averages['avg_answer_accuracy']:.3f}, "
            f"AUROC={averages['avg_auroc']:.3f}, "
            f"AUPRC={averages['avg_auprc']:.3f}, "
            f"F1={averages['avg_f1']:.3f}"
        )
    else:
        print("No explanation metrics recorded for positive samples.")

    samplewise = _compute_samplewise_averages(metrics)
    sample_count = samplewise.pop("sample_count")
    if sample_count > 0:
        print(
            "Per-sample averaged explanation accuracy: "
            f"AnswerAcc={samplewise['answer_accuracy']:.3f}, "
            f"AUROC={samplewise['auroc']:.3f}, "
            f"AUPRC={samplewise['auprc']:.3f}, "
            f"F1={samplewise['f1']:.3f}"
        )

    if aggregated_stability:
        stability_summary = ", ".join(f"{key}={value:.3f}" for key, value in aggregated_stability.items())
        print(f"Edge-mask stability averages: {stability_summary}")

    print(f"Wrote averaged metrics to {output_csv}")


if __name__ == "__main__":
    main()
