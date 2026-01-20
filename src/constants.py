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

# Max new tokens for each subset
MAX_NEW_TOKENS = {
    "node_count": 4,
    "edge_count": 4,
    "cycle_check": 8,
    "triangle_counting": 4,
    "reachability": 4,
    "node_degree": 4,
    "edge_existence": 4,
    "ba_shapes": 8,
    "tree_cycle": 8,
    "tree_grid": 8,
    "ba_two_motifs": 12,
}
EXT_MAX_NEW_TOKENS = {
    "node_count": 96,
    "edge_count": 256,
    "cycle_check": 512,
    "triangle_counting": 256,
}
