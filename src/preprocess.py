import re
from pprint import pprint
from typing import Any, Dict, List, Tuple

import torch


def add_graph_column(example):
    text = example["question"]
    nodes = extract_nodes_from_text(text)
    edges = extract_edges_from_text(text)
    example["graph"] = create_pyg_dict(nodes, edges, node_feat_dim=1)
    example["answer"] = example["answer"].strip()
    return example


def extract_nodes_from_text(text: str) -> List[int]:
    """
    Extracts a list of graph nodes from the given text.
    """
    match = re.search(r"among nodes (.*?)\.", text, re.DOTALL)
    if match:
        nodes_str = match.group(1)
        nodes = [int(n) for n in re.findall(r"\d+", nodes_str)]
        return nodes
    return []


def extract_edges_from_text(text: str) -> List[Tuple[int, int]]:
    """
    Extracts a list of graph edges from the given text.
    """
    # [Note] Added \s* between . and Q: to handle newlines and spaces
    match = re.search(r"The edges in G are: (.*?)\.\s*Q:", text, re.DOTALL)
    if match:
        edges_str = match.group(1)
        edge_pairs_str = re.findall(r"\((\d+),\s*(\d+)\)", edges_str)
        edges = [(int(u), int(v)) for u, v in edge_pairs_str]
        return edges
    return []


def create_pyg_dict(nodes: List[int], edges: List[Tuple[int, int]], node_feat_dim: int = 1) -> Dict[str, Any]:
    """
    Creates a PyG-format graph dictionary from lists of nodes and edges.
    """
    num_nodes = len(nodes)
    x = torch.zeros((num_nodes, node_feat_dim), dtype=torch.float)

    source_nodes = []
    target_nodes = []
    if edges:
        for u, v in edges:
            source_nodes.extend([u, v])
            target_nodes.extend([v, u])

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    return {
        "x": x,
        "edge_index": edge_index,
        "batch": batch,
    }


# --- Execution block ---
if __name__ == "__main__":

    # User-provided text
    text1 = (
        "In an undirected graph, (i,j) means that node i and node j are "
        "connected with an undirected edge. G describes a graph among "
        "nodes 0, 1, 2, 3, 4, 5, 6, and 7.\n"
        "The edges in G are: (0, 1) (0, 5) (0, 6) (0, 7) (1, 2) (1, 4) "
        "(1, 5) (1, 6) (1, 7) (2, 3) (2, 4) (3, 7) (4, 5) (5, 7).\n"
        "Q: How many edges are in this graph?\n"
        "A: "
    )

    text2 = (
        "In an undirected graph, (i,j) means that node i and node j are "
        "connected with an undirected edge. G describes a graph among "
        "nodes 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, and 17.\n"
        "The edges in G are: (0, 1) (0, 2) (0, 3) (0, 5) (0, 7) (0, 9) (0, 10) "
        "(0, 13) (0, 14) (0, 15) (0, 17) (1, 4) (1, 6) (1, 7) (1, 9) (1, 15) "
        "(2, 3) (2, 6) (2, 8) (2, 10) (2, 12) (2, 13) (2, 14) (2, 15) (2, 16) "
        "(2, 17) (3, 4) (3, 5) (3, 6) (3, 8) (3, 11) (3, 12) (3, 13) (3, 14) "
        "(3, 15) (4, 5) (4, 6) (4, 7) (4, 8) (4, 9) (4, 10) (4, 12) (4, 15) "
        "(4, 16) (4, 17) (5, 7) (5, 8) (5, 9) (5, 10) (5, 11) (5, 12) (5, 14) "
        "(5, 15) (5, 16) (5, 17) (6, 7) (6, 8) (6, 9) (6, 10) (6, 11) (6, 12) "
        "(6, 13) (7, 10) (7, 11) (7, 13) (7, 15) (7, 16) (7, 17) (8, 9) (8, 11) "
        "(8, 12) (8, 16) (9, 14) (9, 15) (10, 12) (10, 14) (10, 16) (10, 17) "
        "(11, 12) (12, 14) (12, 15) (12, 16) (12, 17) (13, 14) (13, 15) (13, 16) "
        "(14, 15) (14, 16) (14, 17) (15, 17).\n"
        "Q: How many edges are in this graph?\n"
        "A:"
    )

    print("--- Processing result ---")
    nodes = extract_nodes_from_text(text1)
    edges = extract_edges_from_text(text1)

    if nodes:
        pyg_graph = create_pyg_dict(nodes, edges)
        print(f"✅ Extracted nodes: {nodes}")
        print(f"✅ Extracted edges: {edges}")
        print("\n✅ Generated PyG dictionary:")
        pprint(pyg_graph, sort_dicts=False)
    else:
        print("❌ Failed to extract graph information.")
