import argparse
import os
import sys
from typing import Any, Dict

from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocess import extract_edges_from_text, extract_nodes_from_text  # noqa: E402


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--subset",
        type=str,
        choices=["node_count", "edge_count", "cycle_check", "triangle_counting"],
        default="node_count",
    )
    return p.parse_args()


def extract_cycles(nodes: list[int], edges: list[tuple[int, int]]) -> list[list[int]]:
    if not nodes or not edges:
        return []

    adjacency: dict[int, set[int]] = {node: set() for node in nodes}
    for u, v in edges:
        if u == v:
            continue
        if u in adjacency and v in adjacency:
            adjacency[u].add(v)
            adjacency[v].add(u)

    def canonical_cycle(path: list[int]) -> tuple[int, ...]:
        forward = tuple(path)
        backward = (path[0],) + tuple(reversed(path[1:]))
        return forward if forward <= backward else backward

    cycles: set[tuple[int, ...]] = set()
    for start in sorted(adjacency):
        stack: list[tuple[int, list[int], set[int]]] = [(start, [start], {start})]
        while stack:
            current, path, visited = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor == start and len(path) >= 3:
                    cycles.add(canonical_cycle(path))
                elif neighbor > start and neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], visited | {neighbor}))

    return [list(cycle) for cycle in sorted(cycles)]


def extract_triangles(nodes: list[int], edges: list[tuple[int, int]]) -> list[list[int]]:
    if not nodes or not edges:
        return []

    adjacency: dict[int, set[int]] = {node: set() for node in nodes}
    for u, v in edges:
        if u == v:
            continue
        if u in adjacency and v in adjacency:
            adjacency[u].add(v)
            adjacency[v].add(u)

    triangles: set[tuple[int, int, int]] = set()
    for u in sorted(adjacency):
        for v in adjacency[u]:
            if v <= u:
                continue
            common = adjacency[u] & adjacency[v]
            for w in common:
                if w <= v:
                    continue
                triangles.add(tuple(sorted((u, v, w))))

    return [list(triangle) for triangle in sorted(triangles)]


def modify_columns(example, subset: str):
    # Common column processing
    example["nodes"] = extract_nodes_from_text(example["question"])
    example["edges"] = extract_edges_from_text(example["question"])
    example["nnodes"] = int(example["nnodes"])
    example["nedges"] = int(example["nedges"])

    # Subset-specific processing
    if subset == "cycle_check":
        example["cycles"] = extract_cycles(example["nodes"], example["edges"])
        example["answer"] = "yes" if "yes" in example["answer"].lower() else "no"
    if subset == "triangle_counting":
        example["triangles"] = extract_triangles(example["nodes"], example["edges"])

    return example


def create_dataset(subset: str) -> Dict[str, Any]:
    train_raw = load_dataset("baharef/GraphQA", subset, split="zero_shot_train").select(range(32))
    eval_raw = load_dataset("baharef/GraphQA", subset, split="zero_shot_validation").select(range(32))
    test_raw = load_dataset("baharef/GraphQA", subset, split="zero_shot_test").select(range(32))

    COL_ORDER = ["question", "answer", "task_description", "nodes", "edges", "nnodes", "nedges"] + [
        "cycles" if subset == "cycle_check" else "triangles" if subset == "triangle_counting" else []
    ]
    COL_REMOVE = ["algorithm", "text_encoding"]

    def column_editor(example):
        return modify_columns(example, subset)

    org_datasets = {"train": train_raw, "eval": eval_raw, "test": test_raw}
    new_datasets = {
        k: v.map(
            column_editor, remove_columns=COL_REMOVE, desc="Modifying columns", num_proc=os.cpu_count()
        ).select_columns(COL_ORDER)
        for k, v in org_datasets.items()
    }

    return new_datasets


if __name__ == "__main__":
    args = build_args()
    datasets = create_dataset(args.subset)

    output_dir = os.path.join("dataset", args.subset)
    os.makedirs(output_dir, exist_ok=True)

    for split, ds in datasets.items():
        jsonl_path = os.path.join(output_dir, f"{split}.jsonl")
        ds.to_json(jsonl_path, lines=True, force_ascii=False)
        print(f"[INFO] Exported {split} -> {jsonl_path}")
