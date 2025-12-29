import os
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import torch
from matplotlib import colors as mcolors
from torch_geometric.explain import Explanation


def visualize_motif_explanation(
    sample: Dict[str, object],
    explanation: Explanation,
    graph_path: str,
    exp_accuracy: Dict[str, float],
    ans_accuracy: float,
) -> None:
    """Visualize edge attributions for a MotifQA sample with motif highlights.

    Parameters
    ----------
    sample : dict[str, object]
        Dataset sample that contains ``graph``, ``nodes``, ``motif_nodes``, and ``index`` fields.
        The ``graph`` entry must provide a PyG-compatible dictionary with ``edge_index`` (bidirectional)
        and optionally ``x`` for node features.
    explanation : torch_geometric.explain.Explanation
        Explanation object whose ``edge_mask`` scores are rendered as edge intensities.
    graph_path : str
        Destination path for the rendered SVG figure. Parent directories are created if missing.
    exp_accuracy : dict[str, float]
        Mapping that bundles explanation accuracy metrics; must contain ``f1``,
        ``auroc``, and ``auprc`` keys.
    ans_accuracy : float
        Fraction of successful generations across trials, expected within ``[0, 1]``.

    Returns
    -------
    None
        The function saves the visualization to ``graph_path`` and does not return a value.
    """
    graph_dict = sample["graph"]
    edge_index = torch.as_tensor(graph_dict["edge_index"], dtype=torch.long)

    num_nodes = 0
    if isinstance(graph_dict, dict) and "x" in graph_dict:
        num_nodes = int(torch.as_tensor(graph_dict["x"]).shape[0])
    if edge_index.numel() > 0:
        num_nodes = max(num_nodes, int(edge_index.max().item()) + 1)

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    # Aggregate bidirectional edge weights into undirected edges.
    edge_mask = explanation.edge_mask.detach().cpu().tolist()
    undirected_weights: dict[tuple[int, int], tuple[float, int]] = {}
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(range(num_nodes))
    for src, dst in edge_index.t().tolist():
        layout_graph.add_edge(src, dst)

    for idx, (src, dst) in enumerate(edge_index.t().tolist()):
        key = tuple(sorted((src, dst)))
        weight = float(edge_mask[idx])
        if key in undirected_weights:
            total, count = undirected_weights[key]
            undirected_weights[key] = (total + weight, count + 1)
        else:
            undirected_weights[key] = (weight, 1)
    for (src, dst), (total, count) in undirected_weights.items():
        avg_weight = total / max(count, 1)
        G.add_edge(src, dst, weight=avg_weight)

    pos = nx.spring_layout(layout_graph, seed=42) if len(layout_graph) > 0 else {}

    nodes_list = sample.get("nodes", list(range(num_nodes)))
    node_to_idx = {nid: idx for idx, nid in enumerate(nodes_list)}
    motif_node_ids = sample.get("motif_nodes", [])
    motif_idx_set = {node_to_idx[nid] for nid in motif_node_ids if nid in node_to_idx}

    node_colors = ["#ff8c00" if node in motif_idx_set else "#87ceeb" for node in G.nodes]
    node_sizes = [600 if node in motif_idx_set else 300 for node in G.nodes]
    labels = {idx: str(nodes_list[idx]) if idx < len(nodes_list) else str(idx) for idx in G.nodes}

    edge_weights = [G.edges[edge]["weight"] for edge in G.edges]
    if edge_weights:
        max_weight = max(edge_weights)
        if max_weight > 0:
            norm_weights = [w / max_weight for w in edge_weights]
        else:
            norm_weights = [0.0 for _ in edge_weights]
    else:
        norm_weights = []

    fig, ax_graph = plt.subplots(figsize=(7, 4))

    if len(G) > 0:
        xs = [coord[0] for coord in pos.values()]
        ys = [coord[1] for coord in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        for node, (x_coord, y_coord) in pos.items():
            norm_x = (x_coord - min_x) / span_x
            norm_y = (y_coord - min_y) / span_y
            pos[node] = (0.72 * norm_x + 0.02, 0.75 * norm_y + 0.1)
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax_graph,
            edge_color=norm_weights if norm_weights else None,
            edge_cmap=plt.cm.Blues if norm_weights else None,
            edge_vmin=0.0 if norm_weights else None,
            edge_vmax=1.0 if norm_weights else None,
            width=[1.5 + 3 * w for w in norm_weights] if norm_weights else 1.5,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            ax=ax_graph,
            linewidths=1.5,
            edgecolors="#333333",
        )
        nx.draw_networkx_labels(G, pos, labels=labels, font_color="white", ax=ax_graph)
        if norm_weights:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=mcolors.Normalize(vmin=0.0, vmax=1.0))
            cbar = fig.colorbar(sm, ax=ax_graph, orientation="horizontal", fraction=0.046, pad=0.08)
            cbar.set_label("Edge importance")
    else:
        ax_graph.text(0.5, 0.5, "Empty graph", ha="center", va="center", fontsize=12)

    f1_score = float(exp_accuracy["f1"])
    auroc_score = float(exp_accuracy["auroc"])
    auprc_score = float(exp_accuracy["auprc"])

    textbox_lines = [
        f"Sample #{sample.get('index', 'N/A')}",
        f"Answer Accuracy: {ans_accuracy:.3f}",
        f"F1: {f1_score:.3f}",
        f"AUROC: {auroc_score:.3f}",
        f"AUPRC: {auprc_score:.3f}",
    ]
    ax_graph.text(
        0.98,
        0.98,
        "\n".join(textbox_lines),
        transform=ax_graph.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    ax_graph.set_axis_off()
    fig.tight_layout()
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    fig.savefig(graph_path)
    plt.close(fig)
