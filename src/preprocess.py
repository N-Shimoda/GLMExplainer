import re
from pprint import pprint
from typing import Any, Dict, List, Literal, Tuple

import torch


def extract_nodes_from_text(text: str) -> List[int]:
    """Parse node identifiers from a GraphQA-style question.

    Parameters
    ----------
    text : str
        Question text containing a clause that enumerates nodes.

    Returns
    -------
    list of int
        Ordered list of node IDs. Returns an empty list when no nodes are
        found.
    """
    match = re.search(r"among nodes (.*?)\.", text, re.DOTALL)
    if match:
        nodes_str = match.group(1)
        nodes = [int(n) for n in re.findall(r"\d+", nodes_str)]
        return nodes
    return []


def extract_edges_from_text(text: str) -> List[Tuple[int, int]]:
    """Parse edge pairs from a GraphQA-style question.

    Parameters
    ----------
    text : str
        Question text containing an edge enumeration clause.

    Returns
    -------
    list of tuple of int
        List of undirected edge pairs. Returns an empty list when no edges are
        found.
    """
    # [Note] Added \s* between . and Q: to handle newlines and spaces
    match = re.search(r"The edges in G are: (.*?)\.\s*Q:", text, re.DOTALL)
    if match:
        edges_str = match.group(1)
        edge_pairs_str = re.findall(r"\((\d+),\s*(\d+)\)", edges_str)
        edges = [(int(u), int(v)) for u, v in edge_pairs_str]
        return edges
    return []


def create_pyg_dict(nodes: List[int], edges: List[Tuple[int, int]], lpe_dim: int) -> Dict[str, Any]:
    """
    Create a PyG-format graph dictionary from lists of nodes and edges.

    The node features `x` are initialized using Laplacian Positional Embeddings (LPE):
    - The normalized Laplacian L = I - D^{-1/2} A D^{-1/2} is computed.
    - The eigenvectors corresponding to the smallest eigenvalue (constant vector) are excluded.
    - The top `k` nontrivial eigenvectors are used as node features.
    - Node IDs are mapped to consecutive indices to ensure consistency in `edge_index` and `x`.

    Parameters
    ----------
    nodes : list of int
        List of node IDs extracted from text.
    edges : list of tuple of int
        List of undirected edges (u, v) extracted from text.
    lpe_dim : int
        Number of Laplacian eigenvectors to use for node features.

    Returns
    -------
    dict
        PyG-format dictionary containing:
        - 'x': Node feature matrix (LPE).
        - 'edge_index': Edge indices (bidirectional, consecutive indices).
        - 'batch': Batch vector (all zeros).
    """
    num_nodes = len(nodes)

    # Map node IDs to consecutive indices.
    node_to_idx = {nid: i for i, nid in enumerate(nodes)}

    # Build an undirected adjacency matrix.
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    for u, v in edges:
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            if i == j:
                continue
            A[i, j] = 1.0
            A[j, i] = 1.0

    # Normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.
    if num_nodes == 0:
        x = torch.zeros((0, max(lpe_dim, 0)), dtype=torch.float)
    else:
        deg = A.sum(dim=1)
        inv_sqrt_deg = torch.zeros_like(deg)
        mask = deg > 0
        inv_sqrt_deg[mask] = deg[mask].pow(-0.5)
        D_inv_sqrt = torch.diag(inv_sqrt_deg)
        S = D_inv_sqrt @ A @ D_inv_sqrt
        eye = torch.eye(num_nodes, dtype=torch.float)
        L = eye - S
        L = (L + L.T) / 2  # Symmetrize numerically.

        # Eigen decomposition (ascending); discard the trivial eigenvector.
        if lpe_dim <= 0:
            x = torch.zeros((num_nodes, 0), dtype=torch.float)
        else:
            _, evecs = torch.linalg.eigh(L)
            nontrivial = min(lpe_dim, max(num_nodes - 1, 0))
            x = torch.zeros((num_nodes, lpe_dim), dtype=torch.float)
            if nontrivial > 0:
                x[:, :nontrivial] = evecs[:, 1 : 1 + nontrivial]

    # Build bidirectional edge_index using consecutive indices.
    source_nodes: List[int] = []
    target_nodes: List[int] = []
    if edges:
        for u, v in edges:
            if u in node_to_idx and v in node_to_idx:
                i, j = node_to_idx[u], node_to_idx[v]
                source_nodes.extend([i, j])
                target_nodes.extend([j, i])

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    return {
        "x": x,
        "edge_index": edge_index,
        "batch": batch,
    }


def add_graph_column(example, ds_name: Literal["GraphQA", "MotifQA"], lpe_dim: int = 4) -> Dict[str, Any]:
    """Enrich an example with graph metadata parsed from the question.

    Parameters
    ----------
    example : Mapping[str, Any]
        Input example containing at least ``question``, ``task_description``,
        and ``answer`` fields.
    ds_name : Literal['GraphQA', 'MotifQA']
        Type of dataset to process. Both 'GraphQA' and 'MotifQA' are supported.
    lpe_dim : int, default=4
        Number of Laplacian positional embedding dimensions to include in the
        generated graph features.

    Returns
    -------
    dict
        Updated example with ``prompt``, ``completion``, and ``graph`` keys.
    """
    match ds_name:
        case "GraphQA":
            text = example["question"]
            nodes = extract_nodes_from_text(text)
            edges = extract_edges_from_text(text)
            example["prompt"] = example["task_description"]
            example["completion"] = example["answer"].strip()
            example["graph"] = create_pyg_dict(nodes, edges, lpe_dim=lpe_dim)
        case "MotifQA":
            example["prompt"] = f"Q: {example['prompt']}\nA:"
            example["completion"] = example["response"]
            example["graph"] = create_pyg_dict(example["nodes"], example["edges"], lpe_dim=lpe_dim)

    return example


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
        pyg_graph = create_pyg_dict(nodes, edges, lpe_dim=4)
        print(f"✅ Extracted nodes: {nodes}")
        print(f"✅ Extracted edges: {edges}")
        print("\n✅ Generated PyG dictionary:")
        pprint(pyg_graph, sort_dicts=False)
    else:
        print("❌ Failed to extract graph information.")
