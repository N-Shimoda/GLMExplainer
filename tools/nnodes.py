from datasets import load_dataset

subsets = ["node_count", "edge_count", "cycle_check", "triangle_counting", "maximum_flow"]
splits = ["train", "validation", "test"]
for split in splits:
    for subset in subsets:
        if subset == "maximum_flow" and split == "validation":
            continue
        dataset = load_dataset("baharef/GraphQA", subset, split=f"zero_shot_{split}")
        nnode_list = [int(graph["nnodes"]) for graph in dataset]
        print(f"{subset} ({split})")
        print(min(nnode_list), max(nnode_list), sum(nnode_list) / len(nnode_list))
