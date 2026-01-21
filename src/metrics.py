import re

from src.constants import (
    CLASSIFICATION_SUBSETS,
    GRAPHQA_SUBSETS,
    MOTIFQA_SUBSETS,
    NUMERIC_SUBSETS,
)


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
        The identified class label, or "<|unknown|>" if multiple labels match
        or none match.
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
    if len(preds) != len(refs):
        raise ValueError("The number of predictions and references must be the same.")

    num_unknown = 0

    if subset in NUMERIC_SUBSETS.keys():
        # -100 and -1 indicates references and predictions with no digits, respectively
        ref_digits = [int(matches[-1]) if (matches := re.findall(r"\d+", ref)) else -100 for ref in refs]
        pred_digits = [int(matches[-1]) if (matches := re.findall(r"\d+", pred)) else -1 for pred in preds]
        correct_mask = [False] * len(pred_digits)
        for idx, (ref_d, pred_d) in enumerate(zip(ref_digits, pred_digits)):
            correct_mask[idx] = ref_d == pred_d and not (ref_d < 0 or pred_d < 0)
        acc = sum(correct_mask[: len(ref_digits)]) / max(1, len(refs))

    elif subset in CLASSIFICATION_SUBSETS:
        class_labels = CLASSIFICATION_SUBSETS[subset]
        preds_class = [get_class(class_labels, pred) for pred in preds]
        refs_class = [get_class(class_labels, ref) for ref in refs]
        correct_mask = [False] * len(preds_class)
        for idx, (pred_label, ref_label) in enumerate(zip(preds_class, refs_class)):
            correct_mask[idx] = pred_label == ref_label and pred_label in class_labels
        acc = sum(correct_mask[: len(refs_class)]) / max(1, len(refs_class))
        num_unknown = sum(p not in class_labels for p in preds_class)

    elif subset == "shortest_path":
        # -1 indicates "no path" for this subset
        ref_digits = [int(matches[-1]) if (matches := re.findall(r"\d+", ref)) else -1 for ref in refs]
        pred_digits = [int(matches[-1]) if (matches := re.findall(r"\d+", pred)) else -1 for pred in preds]
        correct_mask = [ref_d == pred_d for ref_d, pred_d in zip(ref_digits, pred_digits)]
        acc = sum(correct_mask[: len(ref_digits)]) / max(1, len(refs))

    else:
        raise NotImplementedError(f"Accuracy computation for subset '{subset}' is not implemented.")

    return acc, num_unknown, correct_mask
