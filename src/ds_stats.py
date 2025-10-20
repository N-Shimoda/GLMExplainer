import json
import os
import statistics as _stats
from typing import Dict, List


def _try_import_numpy():
    try:
        import numpy as _np  # type: ignore

        return _np
    except Exception:
        return None


def _percentile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    _np = _try_import_numpy()
    if _np is not None:
        try:
            return float(_np.percentile(values, q))
        except Exception:
            pass
    arr = sorted(values)
    k = (len(arr) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(arr) - 1)
    if f == c:
        return float(arr[f])
    d0 = arr[f] * (c - k)
    d1 = arr[c] * (k - f)
    return float(d0 + d1)


def _word_count(text) -> int:
    """Return number of whitespace-delimited words for ``text``."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    stripped = text.strip()
    return len(stripped.split()) if stripped else 0


def completion_length_report(ds_dict, subset: str, *, main_process: bool = True) -> Dict[str, dict]:
    """Compute and persist statistics and plots for completion lengths across splits.

    Parameters
    ----------
    ds_dict : datasets.DatasetDict-like
        Dataset dictionary containing multiple splits, each with a ``completion`` column.
    subset : str
        Dataset subset name used for naming the output directory.
    main_process : bool, default=True
        When ``True``, write files and generate plots (prevents duplicate outputs in distributed runs).

    Returns
    -------
    Dict[str, dict]
        Mapping of split name to statistics dictionary.
    """
    # 1) Aggregation: completion lengths per split (measured in words)
    lengths_by_split: Dict[str, List[int]] = {}
    for split in ds_dict.keys():
        ds = ds_dict[split]
        if "completion" not in ds.column_names:
            continue
        lengths = [_word_count(text) for text in ds["completion"]]
        if lengths:
            lengths_by_split[split] = lengths

    # 2) Persist statistics to JSON and generate plots
    stats_payload: Dict[str, dict] = {}
    for split, values in lengths_by_split.items():
        n = len(values)
        std_val = float(_stats.pstdev(values)) if n > 1 else 0.0
        stats_payload[split] = {
            "count": n,
            "min": int(min(values)),
            "p25": _percentile(values, 25),
            "median": float(_stats.median(values)),
            "p75": _percentile(values, 75),
            "max": int(max(values)),
            "mean": float(sum(values) / n),
            "std": std_val,
        }

    if not main_process or not lengths_by_split:
        return stats_payload

    plot_dir = os.path.join("ds_debug", "completion_length", subset)
    os.makedirs(plot_dir, exist_ok=True)

    stats_out = os.path.join(plot_dir, "stats.json")
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump({"subset": subset, "splits": stats_payload}, f, indent=2, ensure_ascii=False)

    # 3) Plotting
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return stats_payload

    # Histogram per split
    for split, lengths in lengths_by_split.items():
        if not lengths:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(lengths, bins=40, color="#4C72B0", edgecolor="black", alpha=0.8)
        plt.title(f"{subset} - {split} completion length distribution (n={len(lengths)})")
        plt.xlabel("Length of completion (words)")
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        out_path = os.path.join(plot_dir, f"{split}_length.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    # Boxplot across all splits
    labels = [k for k, v in lengths_by_split.items() if v]
    data = [lengths_by_split[k] for k in labels]
    if data:
        plt.figure(figsize=(9, 6))
        bp = plt.boxplot(
            data,
            labels=labels,
            showfliers=True,
            showmeans=True,
            meanline=True,
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set(facecolor="#CFE2F3")
        for median in bp["medians"]:
            median.set(color="#D62728", linewidth=2)
        for mean in bp["means"]:
            mean.set(color="#2CA02C", linewidth=2)
        plt.title(f"{subset} - completion length by split")
        plt.ylabel("Length of completion (words)")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, axis="y")
        out_box = os.path.join(plot_dir, "boxplot.png")
        plt.tight_layout()
        plt.savefig(out_box, dpi=150)
        plt.close()

    return stats_payload
