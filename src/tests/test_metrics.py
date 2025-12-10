import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.metrics import comp_accuracy, get_class  # noqa: E402


def test_get_class_returns_single_match():
    assert get_class({"yes", "no"}, "The answer is YES for sure.") == "yes"


def test_get_class_returns_unknown_for_multiple_matches():
    assert get_class({"yes", "no"}, "yes or no, both seem plausible") == "<|unknown|>"


def test_get_class_returns_unknown_when_no_match():
    assert get_class({"house", "cycle"}, "triangle motif not present") == "<|unknown|>"


@pytest.mark.parametrize("subset", ["node_count", "edge_count", "triangle_counting", "node_degree"])
def test_comp_accuracy_numeric_subsets_handle_mixed_correctness(subset):
    preds = ["Count: 5", "No idea", "Third value is 9"]
    refs = ["5.", "7.", "8."]

    acc, unknown, correct_mask = comp_accuracy(preds, refs, subset)

    assert acc == pytest.approx(1 / 3)
    assert unknown == 0
    assert correct_mask == [True, False, False]


@pytest.mark.parametrize(
    "subset",
    ["cycle_check", "reachability", "edge_existence", "ba_shapes", "tree_cycle", "tree_grid"],
)
def test_comp_accuracy_yes_no_subsets_cover_known_unknown_and_ambiguous(subset):
    preds = ["Yes, definitely", "There is NO chance", "yes and no are both mentioned"]
    refs = ["yes", "no", "yes"]

    acc, unknown, correct_mask = comp_accuracy(preds, refs, subset)

    assert acc == pytest.approx(2 / 3)
    assert unknown == 1
    assert correct_mask == [True, True, False]


@pytest.mark.parametrize(
    "subset",
    ["cycle_check", "reachability", "edge_existence", "ba_shapes", "tree_cycle", "tree_grid"],
)
def test_comp_accuracy_yes_no_subsets_all_unknown_when_no_labels(subset):
    preds = ["affirmative response missing", "negative evidence lacking", "undecided outcome"]
    refs = ["yes", "no", "yes"]

    acc, unknown, correct_mask = comp_accuracy(preds, refs, subset)

    assert acc == pytest.approx(0.0)
    assert unknown == 3
    assert correct_mask == [False, False, False]


def test_comp_accuracy_ba_two_motifs_counts_unknowns_and_accuracy():
    preds = ["Found a house motif", "Cycle present", "house and cycle both appear"]
    refs = ["house", "cycle", "cycle"]

    acc, unknown, correct_mask = comp_accuracy(preds, refs, "ba_two_motifs")

    assert acc == pytest.approx(2 / 3)
    assert unknown == 1
    assert correct_mask == [True, True, False]


def test_comp_accuracy_ba_two_motifs_all_unknown_when_no_labels():
    preds = ["motif missing", "uncertain structure", "graph shape unclear"]
    refs = ["house", "cycle", "cycle"]

    acc, unknown, correct_mask = comp_accuracy(preds, refs, "ba_two_motifs")

    assert acc == pytest.approx(0.0)
    assert unknown == 3
    assert correct_mask == [False, False, False]


def test_comp_accuracy_raises_for_unsupported_subset():
    with pytest.raises(ValueError):
        comp_accuracy(["yes"], ["yes"], "unsupported")
