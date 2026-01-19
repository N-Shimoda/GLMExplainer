import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.constants import CLASSIFICATION_SUBSETS  # noqa: E402
from src.metrics import get_class  # noqa: E402

LABEL_ORDER = {
    "cycle_check": ["yes", "no"],
    "reachability": ["yes", "no"],
    "edge_existence": ["yes", "no"],
    "ba_shapes": ["yes", "no"],
    "tree_cycle": ["yes", "no"],
    "tree_grid": ["yes", "no"],
    "ba_two_motifs": ["house", "cycle"],
}
UNKNOWN_LABEL = "<|unknown|>"


def build_args():
    parser = argparse.ArgumentParser(description="Draw a confusion matrix for GraphTokenLM outputs.")
    parser.add_argument(
        "result_file",
        type=str,
        help="Path to a JSON result file under the results/ directory.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=sorted(CLASSIFICATION_SUBSETS.keys()),
        help="Subset name; if omitted, inferred from the results/<subset>/ path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. Defaults to tools/outputs/confusion_<subset>_<stem>.png",
    )
    parser.add_argument("--normalize", action="store_true", help="Normalize confusion matrix by true labels.")
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    return parser.parse_args()


def infer_subset(result_path: Path, results_root: Path) -> str | None:
    try:
        rel = result_path.resolve().relative_to(results_root.resolve())
    except ValueError:
        return None
    return rel.parts[0] if rel.parts else None


def load_rows(result_path: Path) -> list[dict]:
    with result_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "results"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
    raise ValueError(f"Unsupported JSON structure in {result_path}")


def extract_text(row: dict, keys: tuple[str, ...], field_name: str) -> str:
    for key in keys:
        if key in row:
            return str(row[key])
    raise KeyError(f"Missing {field_name} field in row: {row}")


def build_confusion(labels: list[str], preds: list[str], refs: list[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for pred_label, ref_label in zip(preds, refs):
        if ref_label not in label_to_idx or pred_label not in label_to_idx:
            continue
        matrix[label_to_idx[ref_label], label_to_idx[pred_label]] += 1
    return matrix


def main():
    args = build_args()
    repo_root = Path(__file__).resolve().parents[1]
    results_root = repo_root / "results"
    result_path = Path(args.result_file)
    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")
    if results_root not in result_path.resolve().parents:
        raise ValueError(f"Result file must be under {results_root}")

    subset = args.subset or infer_subset(result_path, results_root)
    if subset is None:
        raise ValueError("Subset could not be inferred; pass --subset explicitly.")
    if subset not in CLASSIFICATION_SUBSETS:
        raise ValueError(f"Subset '{subset}' is not a classification subset.")

    rows = load_rows(result_path)
    class_labels = CLASSIFICATION_SUBSETS[subset]
    pred_texts = [extract_text(row, ("preds", "pred", "prediction", "output"), "prediction") for row in rows]
    ref_texts = [extract_text(row, ("answer", "label", "target", "gt"), "answer") for row in rows]
    pred_labels = [get_class(class_labels, pred) for pred in pred_texts]
    ref_labels = [get_class(class_labels, ref) for ref in ref_texts]

    labels = LABEL_ORDER.get(subset, sorted(class_labels))
    if UNKNOWN_LABEL in pred_labels or UNKNOWN_LABEL in ref_labels:
        labels = labels + [UNKNOWN_LABEL]

    matrix = build_confusion(labels, pred_labels, ref_labels)
    display_matrix = matrix.astype(float)
    if args.normalize:
        row_sums = display_matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(
            display_matrix,
            np.where(row_sums == 0, 1, row_sums),
        )

    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=160)
    im = ax.imshow(display_matrix, cmap="Blues")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    threshold = display_matrix.max() * 0.6 if display_matrix.size else 0
    for i in range(display_matrix.shape[0]):
        for j in range(display_matrix.shape[1]):
            val = display_matrix[i, j]
            text_color = "white" if val > threshold else "black"
            if args.normalize:
                label = format(val, ".2f")
            else:
                label = str(int(matrix[i, j]))
            ax.text(j, i, label, ha="center", va="center", color=text_color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    title = f"Confusion Matrix: {subset} ({result_path.stem})"
    if args.normalize:
        title += " (normalized)"
    ax.set_title(title)
    plt.tight_layout()

    output_path = args.output
    if output_path is None:
        out_dir = repo_root / "tools" / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"confusion_{subset}_{result_path.stem}.pdf"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved confusion matrix to {output_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
