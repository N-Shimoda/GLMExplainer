# Grouped bar chart for model results across datasets using matplotlib (no seaborn).
# The figure is also saved to /mnt/data/grouped_bar_chart.png

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, choices=["gnn", "node-feat", "multitask"], default="gnn")
    parser.add_argument("--show-title", action="store_true", help="Whether to show the title on the plot")
    return parser.parse_args()


args = build_args()
match args.exp:
    case "gnn":
        d = {
            "GCN": {"node_count": 97.32, "edge_count": 5.00, "cycle_check": 90.32, "triangle_counting": 18.4},
            "GAT": {"node_count": 20.6, "edge_count": 6.0, "cycle_check": 92.2, "triangle_counting": 17.0},
            "GIN": {"node_count": 6.0, "edge_count": 6.2, "cycle_check": 71.6, "triangle_counting": 10.8},
            "GraphSAGE": {"node_count": 21.4, "edge_count": 5.4, "cycle_check": 72.4, "triangle_counting": 10.6},
            "Zero-Shot": {"node_count": 96.4, "edge_count": 37.0, "cycle_check": 87.8, "triangle_counting": 43.6},
            "QLoRA": {"node_count": 100, "edge_count": 56.2, "cycle_check": 97.2, "triangle_counting": 35.8},
        }
    case "node-feat":
        d = {
            "GCN (LPE only)": {"node_count": 8.2, "edge_count": 8.4, "cycle_check": 90.2, "triangle_counting": 21.4},
            "GCN (LPE + IDX)": {
                "node_count": 97.32,
                "edge_count": 5.00,
                "cycle_check": 90.32,
                "triangle_counting": 18.4,
            },
        }
    case "multitask":
        d = {
            "Normal (LPE only)": {
                "node_count": 25.06,
                "edge_count": 6.54,
                "cycle_check": 72.04,
                "triangle_counting": 19.96,
            },
            "Normal (LPE + IDX)": {
                "node_count": 92.2,
                "edge_count": 14.82,
                "cycle_check": 95.2,
                "triangle_counting": 21.56,
            },
            "Multitask (LPE only)": {
                "node_count": 14.58,
                "edge_count": 6.16,
                "cycle_check": 89.92,
                "triangle_counting": 17.64,
            },
            "Multitask (LPE + IDX)": {
                "node_count": 86.02,
                "edge_count": 21.78,
                "cycle_check": 95.96,
                "triangle_counting": 25.52,
            },
        }
    case _:
        raise ValueError(f"Unknown experiment type: {args.exp}")

# Order datasets and methods
dataset_keys = ["node_count", "edge_count", "cycle_check", "triangle_counting"]
dataset_labels = ["Node Count", "Edge Count", "Cycle Check", "Triangle Counting"]
methods = d.keys()

# Colors and hatches for methods
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
    "Normal (LPE + IDX)": "lightblue",
    "Multitask (LPE only)": "orange",
    "Multitask (LPE + IDX)": "gold",
}
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
if args.show_title:
    plt.title("Accuracy per Subset", fontsize=15)
plt.xticks(x, dataset_labels, fontsize=15)
plt.ylabel("Accuracy (%)", fontsize=15)
plt.ylim(0, 105)
plt.legend(frameon=False, fontsize=12)
plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

plt.tight_layout()

# Save and show
output_path = f"fig/acc_{args.exp}.pdf"
os.makedirs("fig", exist_ok=True)
plt.savefig(output_path, bbox_inches="tight", dpi=480)
