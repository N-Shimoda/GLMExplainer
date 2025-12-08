import re
from typing import Literal


def _normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\.$", "", s)
    return s.lower()


def comp_accuracy(
    preds: list[str],
    refs: list[str],
    subset: Literal[
        "cycle_check",
        "node_count",
        "edge_count",
        "triangle_counting",
        "reachability",
        "house_check",
        "node_degree",
        "edge_existence",
    ],
    exact_match: bool = False,
) -> tuple[float, int, list[bool]]:
    """
    Compute the accuracy of the model's predictions depending on the subset.

    Parameters
    ----------
    preds : list[str]
        The list of model predictions.
    refs : list[str]
        The list of reference answers.
    subset : Literal["cycle_check", "node_count", "edge_count", "triangle_counting", "house_check"]
        The subset of the GraphQA dataset.
    exact_match : bool, default=False
        Whether to use exact match for the "cycle_check" subset.
        (No effect for other subsets.)

    Returns
    -------
    acc : float
        The accuracy of the model's predictions.
    unknowns : int
        The number of unknown predictions.
        This value is only defined for the "cycle_check" subset.
    correct_mask : list[bool]
        Boolean mask indicating whether each prediction is correct.
    """
    if subset not in [
        "cycle_check",
        "node_count",
        "edge_count",
        "triangle_counting",
        "reachability",
        "house_check",
        "node_degree",
        "edge_existence",
    ]:
        raise NotImplementedError(f"Unsupported subset: {subset}")

    match subset:
        case "cycle_check" | "house_check" | "reachability" | "edge_existence":
            if exact_match:
                normalized_preds = [_normalize_text(p) for p in preds]
                normalized_refs = [_normalize_text(r) for r in refs]
                correct_mask = [False] * len(preds)
                for idx, (p_norm, r_norm) in enumerate(zip(normalized_preds, normalized_refs)):
                    correct_mask[idx] = p_norm == r_norm
                acc = sum(correct_mask[: len(refs)]) / max(1, len(refs))
                num_unknown = 0
            else:
                low_preds = [pred.lower() for pred in preds]
                low_refs = [ref.lower() for ref in refs]
                preds_yes_no = ["yes" if "yes" in pred else "no" if "no" in pred else "unknown" for pred in low_preds]
                refs_yes_no = ["yes" if "yes" in ref else "no" if "no" in ref else "unknown" for ref in low_refs]
                correct_mask = [False] * len(preds)
                for idx, (pred_label, ref_label) in enumerate(zip(preds_yes_no, refs_yes_no)):
                    correct_mask[idx] = pred_label == ref_label
                acc = sum(correct_mask[: len(refs_yes_no)]) / max(1, len(refs_yes_no))
                num_unknown = sum(p == "unknown" for p in preds_yes_no)
        case "node_count" | "edge_count" | "triangle_counting" | "node_degree":
            digit_ans_li = [int(matches[-1]) if (matches := re.findall(r"\d+", ref)) else -100 for ref in refs]
            pred_nums = [int(matches[-1]) if (matches := re.findall(r"\d+", pred)) else -1 for pred in preds]
            correct_mask = [False] * len(preds)
            for idx, (ans, pred_val) in enumerate(zip(digit_ans_li, pred_nums)):
                correct_mask[idx] = ans == pred_val and not (ans < 0 or pred_val < 0)
            acc = sum(correct_mask[: len(digit_ans_li)]) / max(1, len(refs))
            num_unknown = 0

    return acc, num_unknown, correct_mask
