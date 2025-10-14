from datasets import load_dataset

# Node Count
node = load_dataset(
    "json",
    data_files={
        "train": "dataset/node_count/train.jsonl",
        "validation": "dataset/node_count/eval.jsonl",  # Register the eval split as validation
        "test": "dataset/node_count/test.jsonl",
    },
)
print(node)

# Edge Count
edge = load_dataset(
    "json",
    data_files={
        "train": "dataset/edge_count/train.jsonl",
        "validation": "dataset/edge_count/eval.jsonl",
        "test": "dataset/edge_count/test.jsonl",
    },
)
print(edge)

REPO_NAME = "naos0919/GraphQA"
node.push_to_hub(REPO_NAME, config_name="node_count", private=True)
edge.push_to_hub(REPO_NAME, config_name="edge_count", private=True)
