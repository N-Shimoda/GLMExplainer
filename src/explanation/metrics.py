from __future__ import annotations

import math
from itertools import combinations

import torch
import torch.nn.functional as F

EDGE_MASK_STABILITY_KEYS = (
    "edge_mask_jaccard",
    "edge_mask_spearman",
    "edge_mask_mean_std",
    "edge_mask_cosine",
)


def _default_stability_metrics() -> dict[str, float]:
    """Return a zero-initialized stability metrics dictionary.

    Returns
    -------
    dict[str, float]
        Mapping from stability metric names to zero.
    """
    return {key: 0.0 for key in EDGE_MASK_STABILITY_KEYS}


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    """Assign ranks to tensor entries handling ties.

    Parameters
    ----------
    values : torch.Tensor
        One-dimensional tensor of scores to rank.

    Returns
    -------
    torch.Tensor
        Tensor of floating-point ranks where ties receive their average rank.
    """
    if values.numel() == 0:
        return values.clone().float()
    sorted_vals, sorted_idx = torch.sort(values)
    ranks = torch.zeros_like(sorted_vals, dtype=torch.float)
    n = sorted_vals.numel()
    i = 0
    while i < n:
        j = i + 1
        while j < n and torch.isclose(sorted_vals[j], sorted_vals[i], rtol=1e-5, atol=1e-8):
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[i:j] = avg_rank
        i = j
    output = torch.zeros_like(ranks)
    output[sorted_idx] = ranks
    return output


def _spearman_rank_correlation(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    """Compute Spearman's rank correlation between two edge masks.

    Parameters
    ----------
    mask_a : torch.Tensor
        First mask vector.
    mask_b : torch.Tensor
        Second mask vector.

    Returns
    -------
    float
        Spearman correlation coefficient. Returns ``nan`` when correlation is undefined.
    """
    if mask_a.numel() == 0 or mask_b.numel() == 0:
        return float("nan")
    ranks_a = _rankdata(mask_a)
    ranks_b = _rankdata(mask_b)
    ranks_a = ranks_a - ranks_a.mean()
    ranks_b = ranks_b - ranks_b.mean()
    denom = ranks_a.norm() * ranks_b.norm()
    if denom.item() == 0.0:
        return float("nan")
    return float((ranks_a @ ranks_b) / denom)


def _topk_jaccard(mask_a: torch.Tensor, mask_b: torch.Tensor, top_k: int = 6) -> float | None:
    """Compute Jaccard index of top-k edges between two masks.

    Parameters
    ----------
    mask_a : torch.Tensor
        First mask vector.
    mask_b : torch.Tensor
        Second mask vector.
    top_k : int, default=6
        Number of top edges to consider as positive class.

    Returns
    -------
    float or None
        Jaccard index in ``[0, 1]`` if the union is non-empty; ``None`` otherwise.
    """
    if mask_a.numel() == 0 or mask_b.numel() == 0 or top_k <= 0:
        return None
    k_a = min(top_k, mask_a.numel())
    k_b = min(top_k, mask_b.numel())
    if k_a == 0 and k_b == 0:
        return None
    top_a = torch.topk(mask_a, k=k_a, largest=True).indices.tolist() if k_a > 0 else []
    top_b = torch.topk(mask_b, k=k_b, largest=True).indices.tolist() if k_b > 0 else []
    if not top_a and not top_b:
        return None
    set_a = set(top_a)
    set_b = set(top_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _cosine_similarity(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    """Compute cosine similarity between two mask vectors.

    Parameters
    ----------
    mask_a : torch.Tensor
        First mask vector.
    mask_b : torch.Tensor
        Second mask vector.

    Returns
    -------
    float
        Cosine similarity in ``[-1, 1]``. Returns ``0.0`` when inputs are empty.
    """
    if mask_a.numel() == 0 or mask_b.numel() == 0:
        return 0.0
    if mask_a.numel() != mask_b.numel():
        min_len = min(mask_a.numel(), mask_b.numel())
        if min_len == 0:
            return 0.0
        mask_a = mask_a[:min_len]
        mask_b = mask_b[:min_len]
    vec_a = mask_a.reshape(1, -1).float()
    vec_b = mask_b.reshape(1, -1).float()
    return float(F.cosine_similarity(vec_a, vec_b, dim=1).item())


def _safe_mean(values: list[float]) -> float:
    """Compute the arithmetic mean with empty-list handling.

    Parameters
    ----------
    values : list[float]
        Sequence of numeric values.

    Returns
    -------
    float
        Mean of ``values`` or ``0.0`` when the input list is empty.
    """
    return float(sum(values) / len(values)) if values else 0.0


def _compute_single_sample_metrics(masks: list[torch.Tensor], *, jaccard_k: int = 6) -> dict[str, float]:
    """Compute edge-mask stability metrics for a single sample.

    Parameters
    ----------
    masks : list[torch.Tensor]
        List of edge masks belonging to the same sample.

    Returns
    -------
    dict[str, float]
        Dictionary containing per-sample Jaccard, Spearman, mean STD, and cosine scores.
    """
    metrics = _default_stability_metrics()
    if not masks:
        return metrics

    lengths = {mask.numel() for mask in masks}
    if not lengths:
        return metrics
    if len(lengths) > 1:
        min_len = min(lengths)
        if min_len == 0:
            return metrics
        masks = [mask[:min_len] for mask in masks]

    mask_stack = torch.stack([mask.float() for mask in masks], dim=0)
    if mask_stack.size(0) > 1:
        std_per_edge = mask_stack.std(dim=0, unbiased=False)
    else:
        std_per_edge = torch.zeros_like(mask_stack[0])
    metrics["edge_mask_mean_std"] = float(std_per_edge.mean().item())

    if len(masks) < 2:
        return metrics

    jaccard_scores: list[float] = []
    spearman_scores: list[float] = []
    cosine_scores: list[float] = []
    for i, j in combinations(range(len(masks)), 2):
        jaccard = _topk_jaccard(masks[i], masks[j], top_k=jaccard_k)
        if jaccard is not None:
            jaccard_scores.append(jaccard)
        spearman = _spearman_rank_correlation(masks[i], masks[j])
        if not math.isnan(spearman):
            spearman_scores.append(spearman)
        cosine_scores.append(_cosine_similarity(masks[i], masks[j]))

    metrics["edge_mask_jaccard"] = _safe_mean(jaccard_scores)
    metrics["edge_mask_spearman"] = _safe_mean(spearman_scores)
    metrics["edge_mask_cosine"] = _safe_mean(cosine_scores)
    return metrics


def compute_edge_mask_stability_metrics_per_sample(
    sample_edge_masks: dict[int, list[torch.Tensor]],
    *,
    jaccard_k: int = 6,
) -> dict[int, dict[str, float]]:
    """Compute per-sample edge-mask stability metrics.

    Parameters
    ----------
    sample_edge_masks : dict[int, list[torch.Tensor]]
        Mapping from sample index to a list of per-trial edge masks.

    Returns
    -------
    dict[int, dict[str, float]]
        Mapping from sample index to its stability metrics.
    """
    per_sample: dict[int, dict[str, float]] = {}
    for sample_idx, masks in sample_edge_masks.items():
        per_sample[sample_idx] = _compute_single_sample_metrics(masks, jaccard_k=jaccard_k)
    return per_sample


def compute_edge_mask_stability_metrics(
    sample_edge_masks: dict[int, list[torch.Tensor]],
    *,
    jaccard_k: int = 6,
) -> dict[str, float]:
    """Compute aggregated edge-mask stability metrics over all samples.

    Parameters
    ----------
    sample_edge_masks : dict[int, list[torch.Tensor]]
        Mapping from sample index to its list of edge masks.

    Returns
    -------
    dict[str, float]
        Dictionary containing dataset-level averages for each stability metric.
    """
    per_sample = compute_edge_mask_stability_metrics_per_sample(sample_edge_masks, jaccard_k=jaccard_k)
    aggregated = _default_stability_metrics()
    if not per_sample:
        return aggregated
    for key in EDGE_MASK_STABILITY_KEYS:
        aggregated[key] = _safe_mean([metrics[key] for metrics in per_sample.values()])
    return aggregated


__all__ = [
    "EDGE_MASK_STABILITY_KEYS",
    "compute_edge_mask_stability_metrics",
    "compute_edge_mask_stability_metrics_per_sample",
]
