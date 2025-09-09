import argparse
from pprint import pprint

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from glm import GraphTokenLM


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
    try:
        graph_str = f"[{example['question'].split('Q:')[0].split('The edges in G are:')[1].split('.')[0].strip()}]"
        example["edge_list"] = graph_str
    except IndexError:
        example["edge_list"] = "[]"

    return example


def train_glm():
    model = GraphTokenLM()
    sft_config = SFTConfig(
        train_batch_size=1,
        eval_batch_size=1,
        num_train_epochs=3,
        learning_rate=0.05,
        lr_scheduler_type="linear",
        logging_steps=1,
        save_steps=1000,
        save_total_limit=2,
        gradient_accumulation_steps=4,
        fp16=True,
        bf16=False,
        optim="lion",
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
    pprint(eval_ds)
    pprint(train_ds)
    pprint(eval_ds)
    pprint(eval_ds)
