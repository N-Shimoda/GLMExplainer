import argparse
import json
import os
from datetime import datetime

import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

import wandb
from eval import comp_accuracy, eval_model
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
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--num_graph_tokens", type=int, default=4)
    p.add_argument("--wandb", action="store_true", help="Use wandb logging")
    p.add_argument("--wandb_project", type=str, default="GraphQA-GLM")
    p.add_argument("--do_eval", action="store_true", help="Run evaluation after training")
    return p.parse_args()


def train_glm(train_ds, eval_ds, output_dir, args):
    glm_cfg = GraphTokenLMConfig(
        llm_name="Qwen/Qwen3-4B-Instruct-2507",
        node_feat_dim=1,
        num_graph_tokens=args.num_graph_tokens,
    )
    model = GraphTokenLM(glm_cfg)

    tokenizer = AutoTokenizer.from_pretrained(glm_cfg.llm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=args.num_epochs,
        learning_rate=0.05,
        lr_scheduler_type="linear",
        logging_steps=10,
        save_strategy="epoch",
        gradient_accumulation_steps=4,
        fp16=True,
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

    # Dataset
    train_raw = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_train")
    eval_raw = load_dataset(
        "baharef/GraphQA",
        args.subset,
        split="zero_shot_validation" if args.subset != "maximum_flow" else "zero_shot_test",
    )
    train_ds = train_raw.map(add_graph_column, desc="add_graph_column(train)")
    eval_ds = eval_raw.map(add_graph_column, desc="add_graph_column(eval)")

    # Training
    model = train_glm(train_ds, eval_ds, output_dir, args)

    # Evaluation
    if is_main_process() and args.do_eval:
        print("***** Evaluation *****")
        # test_raw = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_test")
        # test_ds = test_raw.map(add_graph_column, desc="add_graph_column(test)")
        # results = eval_model(model, test_ds, batch_size=8)
        results = eval_model(model, train_ds, batch_size=8)

        # 評価
        acc, unknowns = comp_accuracy([r["preds"] for r in results], [r["answers"] for r in results], args.subset)
        print(f"Accuracy: {acc * 100:.2f}%")
        if unknowns:
            print("Unknown predictions:")
            for pred in unknowns:
                print(f" - {pred}")

        # 結果を書き出し
        res_path = os.path.join("results", f"{run_name}.json")
        os.makedirs("results", exist_ok=True)
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved results to {res_path}")

    if dist.is_initialized():
        dist.destroy_process_group()
