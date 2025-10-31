from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, List

import torch
import torch.nn.functional as F


def _rankdata(values: torch.Tensor) -> torch.Tensor:
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


def _safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def compute_edge_mask_stability_metrics(sample_edge_masks: Dict[int, List[torch.Tensor]]) -> Dict[str, float]:
    jaccard_scores: list[float] = []
    spearman_scores: list[float] = []
    cosine_scores: list[float] = []
    mean_std_values: list[float] = []

    for masks in sample_edge_masks.values():
        if not masks:
            continue
        lengths = {mask.numel() for mask in masks}
        if not lengths:
            continue
        if len(lengths) > 1:
            min_len = min(lengths)
            if min_len == 0:
                continue
            masks = [mask[:min_len] for mask in masks]
        mask_stack = torch.stack([mask.float() for mask in masks], dim=0)
        if mask_stack.size(0) > 1:
            std_per_edge = mask_stack.std(dim=0, unbiased=False)
        else:
            std_per_edge = torch.zeros_like(mask_stack[0])
        mean_std_values.append(std_per_edge.mean().item())

        if len(masks) < 2:
            continue

        for i, j in combinations(range(len(masks)), 2):
            jaccard = _topk_jaccard(masks[i], masks[j])
            if jaccard is not None:
                jaccard_scores.append(jaccard)
            spearman = _spearman_rank_correlation(masks[i], masks[j])
            if not math.isnan(spearman):
                spearman_scores.append(spearman)
            cosine_scores.append(_cosine_similarity(masks[i], masks[j]))

    return {
        "edge_mask_jaccard": _safe_mean(jaccard_scores),
        "edge_mask_spearman": _safe_mean(spearman_scores),
        "edge_mask_mean_std": _safe_mean(mean_std_values),
        "edge_mask_cosine": _safe_mean(cosine_scores),
    }


__all__ = ["compute_edge_mask_stability_metrics"]
