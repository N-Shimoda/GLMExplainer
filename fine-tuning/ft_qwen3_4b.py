"""
Fine-tune Qwen/Qwen3-4B-Instruct-2507 on GraphQA with QLoRA (4bit).
- Dataset: pick the desired subset/split from baharef/GraphQA.
- Method: TRL SFTTrainer + PEFT (LoRA) + bitsandbytes 4-bit (QLoRA).
- Evaluation: simple exact match (ignores whitespace, newlines, trailing period).
"""

import argparse
import os
import time
from typing import Dict

import torch
from accelerate.utils import set_seed
from datasets import load_dataset
from eval_ft import eval_model
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import wandb
from src.ckpt import _resolve_ckpt_path  # noqa: E402


def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def build_args():
    """
    Parses and returns command-line arguments for fine-tuning a Qwen3-4B model on the GraphQA dataset.

    Returns
    -------
    argparse.Namespace
        An object containing all the parsed command-line arguments.
    """
    # Create parser
    p = argparse.ArgumentParser()
    p.add_argument(
        "--subset",
        choices=["node_count", "edge_count", "cycle_check", "triangle_counting"],
        type=str,
        help="GraphQA subset (see https://huggingface.co/datasets/baharef/GraphQA)",
    )
    p.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-4B-Base",
        help="Base model identifier to load before fine-tuning.",
    )
    p.add_argument("--do-eval", action="store_true", help="Whether to run evaluation after fine-tuning")
    p.add_argument("--wandb", action="store_true", help="Use Weights & Biases for logging")
    p.add_argument("--seed", type=int, default=42)

    # SFT parameters
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--per-device-eval-batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)  # Higher LR is typical for LoRA fine-tuning
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.1)

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    return p.parse_args()


def to_conv_prompt_completion(example: Dict) -> Dict:
    """
    Convert into the conversation-style prompt-completion format expected by TRL SFTTrainer:
      {
        "prompt":    [{"role": "user", "content": "<instruction>"}],
        "completion":[{"role": "assistant", "content": "<answer>"}]
      }
    """
    # return {
    #     "prompt": [
    #         {"role": "system", "content": "You are a careful graph reasoner."},
    #         {"role": "user", "content": example["question"]},
    #     ],
    #     "completion": [{"role": "user", "content": example["answer"]}],
    # }
    return {"prompt": example["question"], "completion": example["answer"].strip()}


def build_dataset(subset: str, do_eval: bool):
    train_raw = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_train")
    eval_raw = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_validation")
    train_ds = train_raw.map(to_conv_prompt_completion, remove_columns=train_raw.column_names)
    eval_ds = eval_raw.map(to_conv_prompt_completion, remove_columns=eval_raw.column_names)

    if do_eval:
        test_ds = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_test")
        test_ds = test_ds.map(to_conv_prompt_completion, remove_columns=test_ds.column_names)
    else:
        test_ds = None

    if is_main_process():
        print(f"First example for training on {subset}:", train_ds[0])

    return train_ds, eval_ds, test_ds


def train_model(train_ds, eval_ds, run_name: str, output_dir: str, base_model: str):
    # 4-bit quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Let Accelerate/DPP run one full model replica per process instead of sharding across GPUs.
    device_map = None
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_map = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, trust_remote_code=True)

    # LoRA configuration (typical projection names for Qwen models)
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # SFT configuration
    sft_cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=10,
        eval_steps=25,
        save_strategy="no",
        packing=True,
        bf16=True,
        optim="adamw_8bit",
        report_to="wandb" if args.wandb else "none",
        completion_only_loss=True,  # Exclude prompt tokens from loss (prompt-completion)
        eos_token=tokenizer.eos_token,
        # Qwen3 ships with a chat template in the tokenizer so it is applied automatically
        # (Optionally set eos_token explicitly: SFTConfig(eos_token=tokenizer.eos_token))
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        peft_config=peft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # Training loop
    trainer.train()

    # Save final model
    final_step = trainer.state.global_step
    trainer.save_model(os.path.join(output_dir, f"checkpoint-{final_step}"))


if __name__ == "__main__":
    args = build_args()
    if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    set_seed(args.seed)

    time_stamp = time.strftime("%m%d_%H%M")
    subset_map = {
        "node_count": "nc",
        "edge_count": "ec",
        "cycle_check": "cc",
        "triangle_counting": "tc",
    }
    subset_abr = subset_map.get(args.subset, "OTHER")
    RUN_NAME = f"{subset_abr}-{time_stamp}"
    OUTPUT_DIR = os.path.join("outputs", args.subset, time_stamp)

    if args.wandb and is_main_process():
        wandb.init(project="GraphQA-ft", name=RUN_NAME)

    # Load GraphQA dataset
    train_ds, eval_ds, test_ds = build_dataset(args.subset, args.do_eval)

    # Fine-tune the model using QLoRA
    if is_main_process():
        print("[INFO] Start training")
    train_model(train_ds, eval_ds, RUN_NAME, OUTPUT_DIR, args.base_model)

    # Evaluate the trained model
    if args.do_eval and is_main_process():
        ckpt_path, _ = _resolve_ckpt_path(OUTPUT_DIR)
        tok = AutoTokenizer.from_pretrained(
            args.base_model, use_fast=False, trust_remote_code=True, padding_side="left"
        )

        # Run evaluation
        print("[INFO] Start evaluation")
        start_time = time.time()
        acc, unknowns = eval_model(ckpt_path, test_ds, args.subset, tok, batch_size=32)
        if args.wandb and is_main_process():
            wandb.log({"test_accuracy": acc, "test_unknown": unknowns})
        print(f"[INFO] Evaluation completed in {time.time() - start_time:.2f} seconds")
