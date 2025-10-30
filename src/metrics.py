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
    subset: Literal["cycle_check", "node_count", "edge_count", "triangle_counting", "house_check"],
    exact_match: bool = False,
) -> tuple[float, int]:
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
    """
    if subset not in ["cycle_check", "node_count", "edge_count", "triangle_counting", "house_check"]:
        raise NotImplementedError(f"Unsupported subset: {subset}")

    match subset:
        case "cycle_check" | "house_check":
            if exact_match:
                acc = sum(_normalize_text(p) == _normalize_text(r) for p, r in zip(preds, refs)) / max(1, len(refs))
                num_unknown = 0
            else:
                low_preds = [pred.lower() for pred in preds]
                low_refs = [ref.lower() for ref in refs]
                preds_yes_no = ["yes" if "yes" in pred else "no" if "no" in pred else "unknown" for pred in low_preds]
                refs_yes_no = ["yes" if "yes" in ref else "no" if "no" in ref else "unknown" for ref in low_refs]
                acc = sum(p == r for p, r in zip(preds_yes_no, refs_yes_no)) / max(1, len(refs_yes_no))
                num_unknown = sum(p == "unknown" for p in preds_yes_no)
        case "node_count" | "edge_count" | "triangle_counting":
            digit_ans_li = [int(matches[-1]) if (matches := re.findall(r"\d+", ref)) else -100 for ref in refs]
            preds = [int(matches[-1]) if (matches := re.findall(r"\d+", pred)) else -1 for pred in preds]
            acc = sum([(ans == pred and not (ans < 0 or pred < 0)) for ans, pred in zip(digit_ans_li, preds)]) / max(
                1, len(refs)
            )
            num_unknown = 0

    return acc, num_unknown
