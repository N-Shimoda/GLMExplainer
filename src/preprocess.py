import re
from typing import Any, Dict, List, Tuple

import torch


def extract_nodes_from_text(text: str) -> List[int]:
    """
    テキストからグラフのノードリストを抽出します。
    """
    match = re.search(r"among nodes (.*?)\.", text, re.DOTALL)
    if match:
        nodes_str = match.group(1)
        nodes = [int(n) for n in re.findall(r"\d+", nodes_str)]
        return nodes
    return []


def extract_edges_from_text(text: str) -> List[Tuple[int, int]]:
    """
    テキストからグラフの辺のリストを抽出します。
    """
    # 【修正点】\. と Q: の間に \s* を追加し、改行やスペースに対応
    match = re.search(r"The edges in G are: (.*?)\.\s*Q:", text, re.DOTALL)
    if match:
        edges_str = match.group(1)
        edge_pairs_str = re.findall(r"\((\d+),\s*(\d+)\)", edges_str)
        edges = [(int(u), int(v)) for u, v in edge_pairs_str]
        return edges
    return []


def create_pyg_dict(nodes: List[int], edges: List[Tuple[int, int]], node_feat_dim: int = 1) -> Dict[str, Any]:
    """
    ノードと辺のリストからPyG形式のグラフ辞書を作成します。
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


# --- 実行ブロック ---
if __name__ == "__main__":

    # ユーザー提供のテキスト
    text1 = (
        "In an undirected graph, (i,j) means that node i and node j are "
        "connected with an undirected edge. G describes a graph among "
        "nodes 0, 1, 2, 3, 4, 5, 6, and 7.\n"
        "The edges in G are: (0, 1) (0, 5) (0, 6) (0, 7) (1, 2) (1, 4) "
        "(1, 5) (1, 6) (1, 7) (2, 3) (2, 4) (3, 7) (4, 5) (5, 7).\n"
        "Q: How many edges are in this graph?\n"
        "A: "
    )

    print("--- 処理結果 ---")
    nodes = extract_nodes_from_text(text1)
    edges = extract_edges_from_text(text1)

    if nodes:
        pyg_graph = create_pyg_dict(nodes, edges)

        print(f"✅ 抽出したノード: {nodes}")
        print(f"✅ 抽出した辺: {edges}")
        print("\n✅ 生成されたPyG辞書:")

        for key, tensor in pyg_graph.items():
            print(f"  '{key}':")
            print(f"    shape: {tensor.shape}")
            print(f"    dtype: {tensor.dtype}")
    else:
        print("❌ グラフ情報の抽出に失敗しました。")
