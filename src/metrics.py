import re

from src.constants import GRAPHQA_SUBSETS, MOTIFQA_SUBSETS


def get_class(class_labels: set[str], pred: str) -> str:
    """
    Determine the class label from the prediction string.

    Parameters
    ----------
    class_labels : set[str]
        A set of possible class labels.
    pred : str
        The model's prediction string.

    Returns
    -------
    str
        The identified class label, "<|unknown|>" if multiple labels match, or
        "unknown" if none match.
    """
    matched_labels = {label for label in class_labels if label in pred.lower()}
    if len(matched_labels) == 1:
        return matched_labels.pop()
    else:
        return "<|unknown|>"


def comp_accuracy(
    preds: list[str],
    refs: list[str],
    subset: str,
) -> tuple[float, int, list[bool]]:
    """
    Compute the accuracy of the model's predictions depending on the subset.

    Parameters
    ----------
    preds : list[str]
        The list of model predictions.
    refs : list[str]
        The list of reference answers.
    subset : str
        The subset name indicating the type of task.

    Returns
    -------
    acc : float
        The accuracy of the model's predictions.
    unknowns : int
        The number of unknown predictions (0 for numeric subsets).
    correct_mask : list[bool]
        Boolean mask indicating whether each prediction is correct.
    """
    if subset not in GRAPHQA_SUBSETS + MOTIFQA_SUBSETS:
        raise ValueError(f"Unknown subset was given: {subset}")

    numeric_subsets = {"node_count": int, "edge_count": int, "triangle_counting": int, "node_degree": int}
    classification_subsets = {
        "cycle_check": {"yes", "no"},
        "reachability": {"yes", "no"},
        "edge_existence": {"yes", "no"},
        "ba_shapes": {"yes", "no"},
        "tree_cycle": {"yes", "no"},
        "tree_grid": {"yes", "no"},
        "ba_two_motifs": {"house", "cycle"},
    }

    if subset in numeric_subsets.keys():
        digit_ans_li = [int(matches[-1]) if (matches := re.findall(r"\d+", ref)) else -100 for ref in refs]
        pred_nums = [int(matches[-1]) if (matches := re.findall(r"\d+", pred)) else -1 for pred in preds]
        correct_mask = [False] * len(preds)
        for idx, (ans, pred_val) in enumerate(zip(digit_ans_li, pred_nums)):
            correct_mask[idx] = ans == pred_val and not (ans < 0 or pred_val < 0)
        acc = sum(correct_mask[: len(digit_ans_li)]) / max(1, len(refs))
        num_unknown = 0

    elif subset in classification_subsets:
        class_labels = classification_subsets[subset]
        preds_class = [get_class(class_labels, pred) for pred in preds]
        refs_class = [get_class(class_labels, ref) for ref in refs]
        correct_mask = [False] * len(preds)
        for idx, (pred_label, ref_label) in enumerate(zip(preds_class, refs_class)):
            correct_mask[idx] = pred_label == ref_label and pred_label in class_labels
        acc = sum(correct_mask[: len(refs_class)]) / max(1, len(refs_class))
        num_unknown = sum(p not in class_labels for p in preds_class)

    else:
        raise ValueError(f"Unknown subset: {subset}")

    return acc, num_unknown, correct_mask
