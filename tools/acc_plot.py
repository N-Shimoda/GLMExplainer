# Grouped bar chart for model results across datasets using matplotlib (no seaborn).
# The figure is also saved to /mnt/data/grouped_bar_chart.png

import os

import matplotlib.pyplot as plt
import numpy as np

# Given data (restructured): Outer keys are model names, inner keys are dataset/subtask names
# d = {
#     "GCN": {"node_count": 97.32, "edge_count": 5.00, "cycle_check": 90.32, "triangle_counting": 18.4},
#     "GIN": {"node_count": 6.0, "edge_count": 6.2, "cycle_check": 71.6, "triangle_counting": 10.8},
#     "GAT": {"node_count": 20.6, "edge_count": 6.0, "cycle_check": 92.2, "triangle_counting": 17.0},
#     "GraphSAGE": {"node_count": 21.4, "edge_count": 5.4, "cycle_check": 72.4, "triangle_counting": 10.6},
#     "Zero-Shot": {"node_count": 96.4, "edge_count": 37.0, "cycle_check": 87.8, "triangle_counting": 43.6},
#     "QLoRA": {"node_count": 100, "edge_count": 56.2, "cycle_check": 97.2, "triangle_counting": 35.8},
# }
# d = {
#     "GCN (LPE only)": {"node_count": 8.2, "edge_count": 8.4, "cycle_check": 90.2, "triangle_counting": 21.4},
#     "GCN (LPE + IDX)": {"node_count": 97.32, "edge_count": 5.00, "cycle_check": 90.32, "triangle_counting": 18.4},
# }
d = {
    "Normal (LPE only)": {"node_count": 8.2, "edge_count": 8.4, "cycle_check": 90.2, "triangle_counting": 21.4},
    "Multitask (LPE only)": {"node_count": 8.8, "edge_count": 4.92, "cycle_check": 92.72, "triangle_counting": 20.72},
    "Normal (LPE + IDX)": {"node_count": 97.32, "edge_count": 5.0, "cycle_check": 90.32, "triangle_counting": 18.4},
    "Multitask (LPE + IDX)": {"node_count": 8.18, "edge_count": 4.58, "cycle_check": 72.66, "triangle_counting": 9.32},
}

# Order datasets and methods
dataset_keys = ["node_count", "edge_count", "cycle_check", "triangle_counting"]
dataset_labels = ["Node Count", "Edge Count", "Cycle Check", "Triangle Counting"]
# methods = ["GCN", "GAT", "GIN", "GraphSAGE", "Zero-Shot", "QLoRA"]
# methods = ["GCN (LPE only)", "GCN (LPE + IDX)"]
methods = d.keys()

# Explicit color mapping (color-blind friendly palette inspired by Tableau/Matplotlib)
# Adjust if you need brand colors.
method_colors = {
    "GCN": "royalblue",
    "GAT": "lightblue",
    "GIN": "orange",
    "GraphSAGE": "gold",
    "Zero-Shot": "lightgray",
    "QLoRA": "gray",
    "GCN (LPE + IDX)": "royalblue",
    "GCN (LPE only)": "lightblue",
    "Normal (LPE only)": "royalblue",
    "Multitask (LPE only)": "lightblue",
    "Normal (LPE + IDX)": "orange",
    "Multitask (LPE + IDX)": "gold",
}

# Optional hatch (pattern) settings per method. Add patterns to distinguish certain methods in grayscale printing.
method_hatches = {
    "GCN": "////",  # diagonal stripes for GCN
    "GCN (LPE + IDX)": "////",  # diagonal stripes for GCN
    "Normal (LPE only)": "////",
}

# Prepare data matrix: rows=datasets, cols=methods
values = np.array([[d[m][dk] for m in methods] for dk in dataset_keys])

# Plot parameters
n_datasets = len(dataset_keys)
n_models = len(methods)
x = np.arange(n_datasets)  # dataset positions
bar_width = 0.12
offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_width

plt.figure(figsize=(9, 5.5), dpi=160)

# Draw grouped bars
for i, m in enumerate(methods):
    color = method_colors.get(m, None)
    hatch = method_hatches.get(m, None)
    plt.bar(
        x + offsets[i],
        values[:, i],
        width=bar_width,
        label=m,
        color=color,
        edgecolor="black",
        linewidth=0.3,
        hatch=hatch,
    )
    # Annotate bars with values
    for xi, yi in zip(x + offsets[i], values[:, i]):
        plt.text(xi, yi + 0.008, f"{yi:.2f}", ha="center", va="bottom", fontsize=8, rotation=0)

# Aesthetics
plt.xticks(x, dataset_labels)
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)
plt.title("Accuracy per Subset")
plt.legend(frameon=False)
plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

plt.tight_layout()

# Save and show
output_path = "fig/acc_plot.png"
os.makedirs("fig", exist_ok=True)
plt.savefig(output_path, bbox_inches="tight", dpi=480)
