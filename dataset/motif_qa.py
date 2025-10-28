import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import (  # noqa E401
    BAGraph,
    ERGraph,
    GridGraph,
    TreeGraph,
)
from torch_geometric.utils import is_undirected, to_networkx

dataset = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=20, num_edges=2),
    # graph_generator=ERGraph(num_nodes=20, edge_prob=0.15),
    # graph_generator=GridGraph(height=4, width=5),
    # graph_generator=TreeGraph(depth=3, branch=2, undirected=True),
    motif_generator="house",
    num_motifs=1,
)

data = dataset[0]
if not is_undirected(data.edge_index):
    raise ValueError("The generated graph is not undirected.")
G = to_networkx(data, to_undirected=True)

# Setup layout
pos = nx.spring_layout(G, seed=42)

# Extract motif nodes from node_mask
node_mask = getattr(data, "node_mask", None)
print(node_mask)
motif_nodes = node_mask.nonzero(as_tuple=True)[0].tolist()
normal_nodes = [n for n in G.nodes if n not in motif_nodes]
print(f"Motif nodes: {motif_nodes}")

# Draw figure
plt.figure(figsize=(6, 6))
nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_color="skyblue", node_size=40)
nx.draw_networkx_nodes(G, pos, nodelist=motif_nodes, node_color="orange", node_size=80)
nx.draw_networkx_edges(G, pos, alpha=0.3)

# Draw node labels (indices). Use black for normal nodes and white for motif nodes
# so labels remain readable on the different node colors.
labels = {n: str(n) for n in G.nodes()}
if normal_nodes:
    nx.draw_networkx_labels(G, pos, labels={n: labels[n] for n in normal_nodes}, font_size=8, font_color="black")
if motif_nodes:
    nx.draw_networkx_labels(G, pos, labels={n: labels[n] for n in motif_nodes}, font_size=8, font_color="white")

plt.title("Graph with Motifs Highlighted")
plt.axis("off")
plt.savefig("dataset/graph_with_motifs.png", bbox_inches="tight", dpi=300)
plt.close()
