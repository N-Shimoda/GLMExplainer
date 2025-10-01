import argparse
import os
from typing import Any, Dict

from datasets import load_dataset

from preprocess import extract_edges_from_text, extract_nodes_from_text


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--subset",
        type=str,
        choices=["node_count", "edge_count", "cycle_check", "triangle_counting"],
        required=True,
    )
    return p.parse_args()


def extract_cycles(nodes: list[int], edges: list[tuple[int, int]]) -> list[list[int]]:
    pass


def extract_triangles(nodes: list[int], edges: list[tuple[int, int]]) -> list[list[int]]:
    pass


def modify_columns(example):
    example["nodes"] = extract_nodes_from_text(example["question"])
    example["edges"] = extract_edges_from_text(example["question"])
    example["nnodes"] = int(example["nnodes"])
    example["nedges"] = int(example["nedges"])
    return example


def create_dataset(subset: str) -> Dict[str, Any]:
    train_raw = load_dataset("baharef/GraphQA", subset, split="zero_shot_train")
    eval_raw = load_dataset("baharef/GraphQA", subset, split="zero_shot_validation")
    test_raw = load_dataset("baharef/GraphQA", subset, split="zero_shot_test")

    COL_ORDER = ["question", "answer", "task_description", "nodes", "edges", "nnodes", "nedges"]

    org_datasets = {"train": train_raw, "eval": eval_raw, "test": test_raw}
    new_datasets = {
        k: v.map(
            modify_columns, remove_columns=["algorithm", "text_encoding"], desc="Modifying columns"
        ).select_columns(COL_ORDER)
        for k, v in org_datasets.items()
    }

    return new_datasets


if __name__ == "__main__":
    args = build_args()
    datasets = create_dataset(args.subset)

    output_dir = os.path.join("GraphQA", args.subset)
    os.makedirs(output_dir, exist_ok=True)

    for split, ds in datasets.items():
        jsonl_path = os.path.join(output_dir, f"{split}.jsonl")
        ds.to_json(jsonl_path, lines=True, force_ascii=False)
        print(f"[INFO] Exported {split} -> {jsonl_path}")
