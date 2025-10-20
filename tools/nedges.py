from datasets import load_dataset

subsets = ["node_count", "edge_count", "cycle_check", "triangle_counting"]
splits = ["train", "validation", "test"]
for split in splits:
    print(f"--- {split} ---")
    for subset in subsets:
        dataset = load_dataset("baharef/GraphQA", subset, split=f"zero_shot_{split}")
        nedge_list = [int(graph["nedges"]) for graph in dataset]
        print(subset)
        print(f"min: {min(nedge_list)}, max: {max(nedge_list)}, avg: {sum(nedge_list) / len(nedge_list)}")
