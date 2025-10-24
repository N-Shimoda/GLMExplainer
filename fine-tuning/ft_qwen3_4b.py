"""
Fine-tune Qwen/Qwen3-4B-Instruct-2507 on GraphQA with QLoRA (4bit).
- Dataset: pick the desired subset/split from baharef/GraphQA.
- Method: TRL SFTTrainer + PEFT (LoRA) + bitsandbytes 4-bit (QLoRA).
- Evaluation: simple exact match (ignores whitespace, newlines, trailing period).
"""

import argparse
import json
import os
import time
from math import ceil
from pprint import pprint
from typing import Dict, Tuple

import torch
from accelerate.utils import set_seed
from datasets import Dataset, load_dataset
from eval_ft import eval_model
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import wandb
from src.ckpt import _resolve_ckpt_path


def is_main_process() -> bool:
    """Check if the current process is the main one (LOCAL_RANK=0)."""
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def build_args():
    """
    Parse CLI flags and split them into config dictionaries for SFT and LoRA.

    Returns
    -------
    sft_args : dict
        Configuration dictionary for SFTTrainer.
    lora_args : dict
        Configuration dictionary for LoRA.
    args : Namespace
        Remaining parsed CLI arguments.
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

    # SFT parameters
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--per-device-eval-batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)  # Higher LR is typical for LoRA fine-tuning
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--save-intermediate-models", action="store_true", help="Save intermediate checkpoints")
    p.add_argument("--save-interval-epochs", type=int, default=1, help="Save intermediate checkpoints every N epochs")

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    parsed_args = p.parse_args()

    sft_args = {
        "num_train_epochs": parsed_args.epochs,
        "per_device_train_batch_size": parsed_args.per_device_train_batch_size,
        "per_device_eval_batch_size": parsed_args.per_device_eval_batch_size,
        "gradient_accumulation_steps": parsed_args.grad_accum_steps,
        "learning_rate": parsed_args.lr,
        "warmup_ratio": parsed_args.warmup_ratio,
        "weight_decay": parsed_args.weight_decay,
        "save_intermediate_models": parsed_args.save_intermediate_models,
        "save_interval_epochs": parsed_args.save_interval_epochs,
    }
    lora_args = {
        "r": parsed_args.lora_r,
        "lora_alpha": parsed_args.lora_alpha,
        "lora_dropout": parsed_args.lora_dropout,
    }

    added_attrs = sft_args.keys() | lora_args.keys() | set(["epochs", "grad_accum_steps", "lr", "lora_r"])
    for attr in added_attrs:
        if hasattr(parsed_args, attr):
            delattr(parsed_args, attr)

    return sft_args, lora_args, parsed_args


def build_run_context(subset: str) -> tuple[str, str]:
    time_stamp = time.strftime("%m%d_%H%M")
    subset_map = {
        "node_count": "nc",
        "edge_count": "ec",
        "cycle_check": "cc",
        "triangle_counting": "tc",
    }
    subset_abr = subset_map.get(subset, "OTHER")
    run_name = f"{subset_abr}-{time_stamp}"
    output_dir = os.path.join("outputs", subset, time_stamp)
    os.makedirs(output_dir, exist_ok=True)
    return run_name, output_dir


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


def build_dataset(subset: str, do_eval: bool) -> Tuple[Dataset, Dataset, Dataset | None]:
    """
    Load and preprocess the specified GraphQA subset.

    Parameters
    ----------
    subset : str
        One of "node_count", "edge_count", "cycle_check", "triangle_counting".
    do_eval : bool
        Whether to load the test split for evaluation.

    Returns
    -------
    train_ds : Dataset
        Training dataset.
    eval_ds : Dataset
        Evaluation dataset.
    test_ds : Dataset | None
        Test dataset if do_eval is True, otherwise None.
    """
    train_raw = load_dataset("baharef/GraphQA", subset, split="zero_shot_train")
    eval_raw = load_dataset("baharef/GraphQA", subset, split="zero_shot_validation")
    train_ds = train_raw.map(to_conv_prompt_completion, remove_columns=train_raw.column_names)
    eval_ds = eval_raw.map(to_conv_prompt_completion, remove_columns=eval_raw.column_names)

    if do_eval:
        test_ds = load_dataset("baharef/GraphQA", subset, split="zero_shot_test")
        # rm_cols = [col for col in test_ds.column_names if col not in ["question", "answer"]]
        # test_ds = test_ds.map(to_conv_prompt_completion, remove_columns=rm_cols)
    else:
        test_ds = None

    if is_main_process():
        print(f"[INFO] First training example for {subset}:")
        pprint(train_ds[0])

    return train_ds, eval_ds, test_ds


def train_model(train_ds, eval_ds, output_dir: str, sft_args: dict, lora_args: dict, args):
    """
    Fine-tune the model using QLoRA (4-bit quantization + LoRA).

    Parameters
    ----------
    train_ds : Dataset
        Training dataset.
    eval_ds : Dataset
        Evaluation dataset.
    output_dir : str
        Directory to save the fine-tuned model and checkpoints.
    sft_args : dict
        Configuration dictionary for SFTTrainer.
    lora_args : dict
        Configuration dictionary for LoRA.
    args : Namespace
        Parsed CLI arguments.
    """
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
        args.base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, trust_remote_code=True)

    # LoRA configuration (typical projection names for Qwen models)
    peft_cfg = LoraConfig(
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        **lora_args,
    )

    # Compute save interval steps
    save_intermediate_models = sft_args.pop("save_intermediate_models")
    save_interval_epochs = sft_args.pop("save_interval_epochs")

    # SFT configuration
    sft_cfg = SFTConfig(
        completion_only_loss=True,  # Exclude prompt tokens from loss (prompt-completion)
        eos_token=tokenizer.eos_token,
        packing=True,
        bf16=True,
        optim="adamw_8bit",
        # Logging and saving
        output_dir=output_dir,
        logging_steps=5,
        eval_strategy="steps",
        save_strategy="steps" if save_intermediate_models else "no",
        load_best_model_at_end=True,
        metric_for_best_model="eval_ppl",
        greater_is_better=False,
        report_to="wandb" if args.wandb else "none",
        **sft_args,
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

    train_dataloader = trainer.get_train_dataloader()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    micro_batches_per_epoch = len(train_dataloader)
    steps_per_epoch = ceil(micro_batches_per_epoch / trainer.args.gradient_accumulation_steps)

    if save_intermediate_models:
        trainer.args.save_steps = steps_per_epoch * save_interval_epochs
        trainer.args.eval_steps = trainer.args.save_steps

    if is_main_process():
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "len(train_ds)": len(train_ds),
                    "micro_batches_per_epoch": micro_batches_per_epoch,
                    "steps_per_epoch": steps_per_epoch,
                    "save_steps": trainer.args.save_steps,
                },
                indent=4,
            )
        )

    # Training loop
    trainer.train()


if __name__ == "__main__":
    sft_args, lora_args, args = build_args()
    RUN_NAME, OUTPUT_DIR = build_run_context(args.subset)

    if args.wandb and is_main_process():
        wandb.init(project="GraphQA-ft", name=RUN_NAME)

    set_seed(42)
    if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Load GraphQA dataset
    train_ds, eval_ds, test_ds = build_dataset(args.subset, args.do_eval)

    # Fine-tune the model using QLoRA
    if is_main_process():
        print("[INFO] Start training")
    train_model(train_ds, eval_ds, OUTPUT_DIR, sft_args, lora_args, args)

    # Evaluate the trained model
    if args.do_eval:
        ckpt_path, _ = _resolve_ckpt_path(OUTPUT_DIR)
        tok = AutoTokenizer.from_pretrained(
            args.base_model, use_fast=False, trust_remote_code=True, padding_side="left"
        )

        if is_main_process():
            print("[INFO] Start evaluation")
        start_time = time.time()
        acc, unknowns = eval_model(ckpt_path, test_ds, args.subset, tok, batch_size=32)

        if is_main_process():
            print(f"[INFO] Evaluation completed in {time.time() - start_time:.2f} seconds")
        if args.wandb and is_main_process():
            wandb.log({"test_accuracy": acc, "test_unknown": unknowns})
