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

    nodes_sorted = sorted(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes_sorted)}
    n = len(nodes_sorted)

    adjacency_mask = [0] * n
    for u, v in edges:
        if u == v:
            continue
        if u not in node_to_idx or v not in node_to_idx:
            continue
        ui = node_to_idx[u]
        vi = node_to_idx[v]
        adjacency_mask[ui] |= 1 << vi
        adjacency_mask[vi] |= 1 << ui

    def canonical_cycle(path: list[int]) -> tuple[int, ...]:
        forward = tuple(path)
        backward = (path[0],) + tuple(reversed(path[1:]))
        return forward if forward <= backward else backward

    cycles: set[tuple[int, ...]] = set()
    for start_idx, start in enumerate(nodes_sorted):
        neighbor_mask = adjacency_mask[start_idx]
        if neighbor_mask == 0:
            continue

        path: list[int] = [start]
        visited_mask = 1 << start_idx

        def dfs(current_idx: int) -> None:
            nonlocal visited_mask
            mask = adjacency_mask[current_idx]
            while mask:
                lsb = mask & -mask
                mask ^= lsb
                neighbor_idx = lsb.bit_length() - 1
                neighbor_id = nodes_sorted[neighbor_idx]
                if neighbor_idx == start_idx and len(path) >= 3:
                    cycles.add(canonical_cycle(path))
                elif neighbor_id > start and not (visited_mask & (1 << neighbor_idx)):
                    visited_mask |= 1 << neighbor_idx
                    path.append(neighbor_id)
                    dfs(neighbor_idx)
                    path.pop()
                    visited_mask &= ~(1 << neighbor_idx)

        dfs(start_idx)

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
    train_raw = load_dataset("baharef/GraphQA", subset, split="zero_shot_train")
    eval_raw = load_dataset("baharef/GraphQA", subset, split="zero_shot_validation")
    test_raw = load_dataset("baharef/GraphQA", subset, split="zero_shot_test")

    base_columns = ["question", "answer", "task_description", "nodes", "edges", "nnodes", "nedges"]
    extra_columns = ["cycles"] if subset == "cycle_check" else ["triangles"] if subset == "triangle_counting" else []
    COL_ORDER = base_columns + extra_columns
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
