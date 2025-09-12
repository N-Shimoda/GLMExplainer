import argparse
from pprint import pprint

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from rev_glm import GraphTokenLM
from src.preprocess import (
    create_pyg_dict,
    extract_edges_from_text,
    extract_nodes_from_text,
)


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--subset",
        type=str,
        choices=["node_count", "edge_count", "cycle_check", "triangle_counting", "maximum_flow"],
        required=True,
    )
    return p.parse_args()


def add_graph_column(example):
    text = example["question"]
    nodes = extract_nodes_from_text(text)
    edges = extract_edges_from_text(text)
    example["graph"] = create_pyg_dict(nodes, edges, node_feat_dim=1)
    return example


def train_glm():
    model = GraphTokenLM()
    sft_config = SFTConfig(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        learning_rate=0.05,
        lr_scheduler_type="linear",
        logging_steps=1,
        save_steps=1000,
        save_total_limit=2,
        gradient_accumulation_steps=4,
        fp16=True,
        bf16=False,
        optim="lion_32bit",
        report_to="none",
        dataset_text_field="task_description",
    )
    trainer = SFTTrainer(model, args=sft_config, train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.train()


if __name__ == "__main__":
    args = build_args()
    print(f"Subset: {args.subset}")

    train_ds = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_train")
    eval_ds = load_dataset(
        "baharef/GraphQA",
        args.subset,
        split="zero_shot_validation" if args.subset != "maximum_flow" else "zero_shot_test",
    )

    train_ds = train_ds.map(add_graph_column)
    eval_ds = eval_ds.map(add_graph_column)
    pprint(train_ds)

    # print("Training example:")
    # pprint(train_ds[0])

    print("Start training...")
    train_glm()
