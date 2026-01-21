# GraphQA and MotifQA subsets
GRAPHQA_SUBSETS = [
    "node_count",
    "edge_count",
    "cycle_check",
    "triangle_counting",
    "reachability",
    "node_degree",
    "edge_existence",
]
MOTIFQA_SUBSETS = ["ba_shapes", "tree_cycle", "tree_grid", "ba_two_motifs", "shortest_path"]

# Mapping of subsets to their output types
NUMERIC_SUBSETS = {
    "node_count": int,
    "edge_count": int,
    "triangle_counting": int,
    "node_degree": int,
}
CLASSIFICATION_SUBSETS = {
    "cycle_check": {"yes", "no"},
    "reachability": {"yes", "no"},
    "edge_existence": {"yes", "no"},
    "ba_shapes": {"yes", "no"},
    "tree_cycle": {"yes", "no"},
    "tree_grid": {"yes", "no"},
    "ba_two_motifs": {"house", "cycle"},
}
