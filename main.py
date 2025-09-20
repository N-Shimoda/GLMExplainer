import argparse
import os
from pprint import pprint

import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

import wandb
from glm import GraphTokenLM
from src.collator import GraphQACollator
from src.preprocess import (
    create_pyg_dict,
    extract_edges_from_text,
    extract_nodes_from_text,
)


def is_main_process() -> bool:
    # torchrun / accelerate で RANK=0 がメイン
    return int(os.environ.get("RANK", "0")) == 0


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--subset",
        type=str,
        choices=["node_count", "edge_count", "cycle_check", "triangle_counting", "maximum_flow"],
        default="edge_count",
    )
    p.add_argument("--num_graph_tokens", type=int, default=4)
    p.add_argument("--wandb", action="store_true", help="Use wandb logging")
    p.add_argument("--wandb_project", type=str, default="GraphQA-GLM")
    return p.parse_args()


def add_graph_column(example):
    text = example["question"]
    nodes = extract_nodes_from_text(text)
    edges = extract_edges_from_text(text)
    example["graph"] = create_pyg_dict(nodes, edges, node_feat_dim=1)
    return example


def train_glm(train_ds, eval_ds, args):
    model = GraphTokenLM(node_feat_dim=1, num_graph_tokens=args.num_graph_tokens)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_config = SFTConfig(
        output_dir="outputs",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        learning_rate=0.05,
        lr_scheduler_type="linear",
        logging_steps=10,
        save_strategy="epoch",
        gradient_accumulation_steps=4,
        fp16=True,
        bf16=False,
        optim="lion_32bit",
        report_to="wandb" if args.wandb else "none",
        dataset_text_field="task_description",
        remove_unused_columns=False,
        ddp_backend="nccl",  # DDP
    )

    collator = GraphQACollator(
        tokenizer=tokenizer,
        text_field="task_description",
        max_length=512,
        num_graph_tokens=args.num_graph_tokens,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    if is_main_process():
        print("***** Training *****")
    trainer.train()
    if is_main_process():
        print("***** Done *****")


if __name__ == "__main__":
    args = build_args()
    if is_main_process():
        print(f"Subset: {args.subset}")
    if args.wandb and is_main_process():
        wandb.init(project=args.wandb_project, name=f"GraphQA-GLM-{args.subset}")

    # 各プロセスで同じデータをロードしてOK（TrainerがSamplerをDDP用に設定）
    train_ds = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_train")
    eval_ds = load_dataset(
        "baharef/GraphQA",
        args.subset,
        split="zero_shot_validation" if args.subset != "maximum_flow" else "zero_shot_test",
    )

    train_ds = train_ds.map(add_graph_column, desc="add_graph_column(train)")
    eval_ds = eval_ds.map(add_graph_column, desc="add_graph_column(eval)")

    if is_main_process():
        pprint(train_ds)

    if is_main_process():
        print("Start training...")
    train_glm(train_ds, eval_ds, args)

    if dist.is_initialized():
        dist.destroy_process_group()
