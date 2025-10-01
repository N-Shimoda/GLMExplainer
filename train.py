import argparse
import os
from datetime import datetime

import torch.distributed as dist
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

import wandb
from datasets import load_dataset
from eval import collect_result, eval_model
from src.collator import GraphQACollator
from src.glm import GraphTokenLM, GraphTokenLMConfig
from src.preprocess import add_graph_column


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

    # Model architecture
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--gnn-type", type=str, default="GCN", choices=["GCN", "GAT", "GIN", "GraphSAGE"])
    p.add_argument("--num-graph-tokens", type=int, default=4)
    p.add_argument("--node-feat-dim", type=int, default=8)
    p.add_argument("--node-pos-emb-dim", type=int, default=8)
    p.add_argument("--gnn-hidden-dim", type=int, default=128)
    p.add_argument("--gnn-out-dim", type=int, default=128)
    p.add_argument("--num-gnn-layers", type=int, default=2)

    # Training parameters
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.01)

    # Logging
    p.add_argument("--wandb", action="store_true", help="Use wandb logging")
    p.add_argument("--wandb-project", type=str, default="GraphQA-GLM")
    p.add_argument("--do-eval", action="store_true", help="Run evaluation after training")
    return p.parse_args()


def build_dataset(subset: str, do_eval: bool = False):
    def modify_dataset(example):
        return add_graph_column(example, k=args.node_feat_dim)

    cols = ["algorithm", "answer", "nedges", "nnodes", "question", "task_description", "text_encoding"]

    train_raw = load_dataset("baharef/GraphQA", subset, split="zero_shot_train")
    eval_raw = load_dataset(
        "baharef/GraphQA",
        subset,
        split="zero_shot_validation" if subset != "maximum_flow" else "zero_shot_test",
    )
    train_ds = train_raw.map(modify_dataset, remove_columns=cols, desc="Preprocessing train")
    eval_ds = eval_raw.map(modify_dataset, remove_columns=cols, desc="Preprocessing eval")

    if do_eval:
        test_raw = load_dataset("baharef/GraphQA", subset, split="zero_shot_test")
        test_ds = test_raw.map(modify_dataset, remove_columns=cols, desc="Preprocessing test")
    else:
        test_ds = None

    DS_DIR = "datasets"
    train_ds.to_json(os.path.join(DS_DIR, "train_ds.jsonl"))
    eval_ds.to_json(os.path.join(DS_DIR, "eval_ds.jsonl"))
    if do_eval:
        test_ds.to_json(os.path.join(DS_DIR, "test_ds.jsonl"))

    return train_ds, eval_ds, test_ds


def train_glm(train_ds, eval_ds, output_dir, args):
    glm_cfg = GraphTokenLMConfig(
        llm_name=args.base_model,
        gnn_type=args.gnn_type,
        node_feat_dim=args.node_feat_dim,
        node_pos_emb_dim=args.node_pos_emb_dim,
        gnn_hidden=args.gnn_hidden_dim,
        gnn_out=args.gnn_out_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_graph_tokens=args.num_graph_tokens,
        num_max_nodes=20 * args.per_device_train_batch_size,
    )
    model = GraphTokenLM(glm_cfg)

    tokenizer = AutoTokenizer.from_pretrained(glm_cfg.llm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = GraphQACollator(
        tokenizer=tokenizer,
        max_length=512,
        num_graph_tokens=args.num_graph_tokens,
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=2,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        logging_steps=10,
        save_strategy="no",
        gradient_accumulation_steps=4,
        bf16=True,
        optim="lion_32bit",
        report_to="wandb" if args.wandb else "none",
        completion_only_loss=True,
        remove_unused_columns=False,
        ddp_backend="nccl",  # DDP
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    if is_main_process():
        print("***** Training *****")
    trainer.train()
    if is_main_process():
        final_step = trainer.state.global_step
        final_ckpt_dir = os.path.join(output_dir, f"checkpoint-{final_step}")
        trainer.save_model(final_ckpt_dir)
        trainer.save_state()
        print("***** Done *****")

    return model


if __name__ == "__main__":
    args = build_args()
    if is_main_process():
        print(f"Subset: {args.subset}")

    date_str = datetime.now().strftime("%m%d-%H%M")
    run_name = f"{args.subset}_{date_str}"
    output_dir = os.path.join("outputs", args.subset, date_str)

    if args.wandb and is_main_process():
        wandb.init(project=args.wandb_project, name=run_name)

    # Training
    train_ds, eval_ds, test_ds = build_dataset(args.subset, do_eval=args.do_eval)
    model = train_glm(train_ds, eval_ds, output_dir, args)

    # Evaluation
    if is_main_process() and args.do_eval:
        print("***** Evaluation *****")
        results = eval_model(model, test_ds, batch_size=8, subset=args.subset)
        res_file = os.path.join("results", args.subset, f"{date_str}.json")
        acc = collect_result(results, res_file, args.subset)
        if args.wandb:
            wandb.log({"test_acc": acc})

    if dist.is_initialized():
        dist.destroy_process_group()
