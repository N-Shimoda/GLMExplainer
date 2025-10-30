import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.metrics import _normalize_text, comp_accuracy  # noqa: E402


def test_normalize_text_strips_whitespace_and_punctuation():
    assert _normalize_text(" Yes.\n\t") == "yes"


def test_comp_accuracy_cycle_check_exact_match():
    preds = [" Yes.\n", "random output"]
    refs = ["yes", "different"]

    acc, unknown, correct_mask = comp_accuracy(preds, refs, "cycle_check", exact_match=True)

    assert acc == pytest.approx(0.5)
    assert unknown == 0
    assert correct_mask == [True, False]


def test_comp_accuracy_cycle_check_keyword_matching_handles_unknowns():
    preds = [
        "This graph has YES cycles!",
        "No cycle detected anywhere.",
        "Cycle status: unclear",
    ]
    refs = [
        "Yes, definitely",
        "There is no cycle",
        "Result pending",
    ]

    acc, unknown, correct_mask = comp_accuracy(preds, refs, "cycle_check")

    assert acc == pytest.approx(1.0)
    assert unknown == 1
    assert correct_mask == [True, True, True]


@pytest.mark.parametrize("subset", ["node_count", "edge_count", "triangle_counting"])
def test_comp_accuracy_numeric_subsets_extracts_digits(subset):
    preds = ["The answer is 5.", "No idea"]
    refs = ["5.", "7."]

    acc, unknown, correct_mask = comp_accuracy(preds, refs, subset)

    assert acc == pytest.approx(0.5)
    assert unknown == 0
    assert correct_mask == [True, False]


def test_comp_accuracy_node_count_matches_numeric_strings():
    preds = ["Total nodes: 12", "Second graph nodes: 0"]
    refs = ["12.", "0."]

    acc, unknown, correct_mask = comp_accuracy(preds, refs, "node_count")

    assert acc == pytest.approx(1.0)
    assert unknown == 0
    assert correct_mask == [True, True]


def test_comp_accuracy_edge_count_handles_multi_digit_refs():
    preds = ["Edges counted: 77", "Residual edges: 3"]
    refs = ["77.", "3."]

    acc, unknown, correct_mask = comp_accuracy(preds, refs, "edge_count")

    assert acc == pytest.approx(1.0)
    assert unknown == 0
    assert correct_mask == [True, True]


def test_comp_accuracy_triangle_counting_zero_and_positive():
    preds = ["Triangles discovered: 0", "Triangles total: 9"]
    refs = ["0.", "9."]

    acc, unknown, correct_mask = comp_accuracy(preds, refs, "triangle_counting")

    assert acc == pytest.approx(1.0)
    assert unknown == 0
    assert correct_mask == [True, True]


def test_comp_accuracy_raises_for_unsupported_subset():
    with pytest.raises(NotImplementedError):
        comp_accuracy(["yes"], ["yes"], "unsupported")
