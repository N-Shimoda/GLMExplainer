import os
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.explain import Explanation


def visualize_motif_explanation(
    sample: Dict[str, object],
    explanation: Explanation,
    graph_path: str,
    f1: float,
    auroc: float,
    auprc: float,
    normal_accuracy_display: str,
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
    f1 : float
        F1 score of the explanation that is reported in the annotation textbox.
    auroc : float
        AUROC metric for the explanation shown in the annotation textbox.
    auprc : float
        AUPRC metric for the explanation shown in the annotation textbox.
    normal_accuracy_display : str
        Textual summary of generation accuracy across trials (for example ``\"2/3\"``) displayed
        alongside the metrics.

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
    undirected_weights: dict[tuple[int, int], float] = {}
    for idx, (src, dst) in enumerate(edge_index.t().tolist()):
        key = tuple(sorted((src, dst)))
        weight = float(edge_mask[idx])
        if key in undirected_weights:
            undirected_weights[key] = max(undirected_weights[key], weight)
        else:
            undirected_weights[key] = weight
    for (src, dst), weight in undirected_weights.items():
        G.add_edge(src, dst, weight=weight)

    pos = nx.spring_layout(G, seed=42) if len(G) > 0 else {}

    nodes_list = sample.get("nodes", list(range(num_nodes)))
    node_to_idx = {nid: idx for idx, nid in enumerate(nodes_list)}
    motif_node_ids = sample.get("motif_nodes", [])
    motif_idx_set = {node_to_idx[nid] for nid in motif_node_ids if nid in node_to_idx}

    node_colors = ["#ff8c00" if node in motif_idx_set else "#000000" for node in G.nodes]
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

    fig, ax = plt.subplots(figsize=(6, 4))
    if len(G) > 0:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
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
            ax=ax,
            linewidths=1.5,
            edgecolors="#333333",
        )
        nx.draw_networkx_labels(G, pos, labels=labels, font_color="white", ax=ax)
    else:
        ax.text(0.5, 0.5, "Empty graph", ha="center", va="center", fontsize=12)

    textbox_lines = [
        f"Sample #{sample.get('index', 'N/A')}",
        f"Normal Accuracy: {normal_accuracy_display}",
        f"F1: {f1:.3f}",
        f"AUROC: {auroc:.3f}",
        f"AUPRC: {auprc:.3f}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(textbox_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    ax.set_axis_off()
    fig.tight_layout()
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    fig.savefig(graph_path, format="svg")
    plt.close(fig)
