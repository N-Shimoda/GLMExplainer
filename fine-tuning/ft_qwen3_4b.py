"""
Qwen/Qwen3-4B-Instruct-2507 を GraphQA で QLoRA (4bit) 微調整
- データ: baharef/GraphQA から subset/split を指定
- 方式: TRL SFTTrainer + PEFT(LoRA) + bitsandbytes 4bit (QLoRA)
- 評価: 簡易 Exact Match（空白/改行/末尾ピリオド無視）
"""

import argparse
import os
import time
from typing import Dict

import torch
import wandb
from accelerate.utils import set_seed
from datasets import load_dataset
from eval import eval_model
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def build_args():
    """
    Parses and returns command-line arguments for fine-tuning a Qwen3-4B model on the GraphQA dataset.

    Returns
    -------
    argparse.Namespace
        An object containing all the parsed command-line arguments.
    """
    DEFAULT_SUBSET = "cycle_check"

    # Create parser
    p = argparse.ArgumentParser()
    p.add_argument(
        "--subset",
        type=str,
        default=DEFAULT_SUBSET,
        help="GraphQA subset（see https://huggingface.co/datasets/baharef/GraphQA）",
    )
    p.add_argument("--output_dir", type=str, default=f"qwen3-4b-{DEFAULT_SUBSET}")
    p.add_argument("--wandb", action="store_true", help="Use Weights & Biases for logging")
    p.add_argument("--seed", type=int, default=42)

    # flow
    p.add_argument("--do_eval", action="store_true", help="Whether to run evaluation after fine-tuning")

    # Hyperparameters (general)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)  # LoRA なので大きめ
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.1)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    return p.parse_args()


def to_conv_prompt_completion(example: Dict) -> Dict:
    """
    TRL SFTTrainer が理解する「会話型 prompt-completion」形式に変換
      {
        "prompt":    [{"role": "user", "content": "<指示>"}],
        "completion":[{"role": "assistant", "content": "<解答>"}]
      }
    """
    # assistant = example["answer"].strip()
    # return {
    #     "prompt": [
    #         {"role": "system", "content": SYS_INST},
    #         {"role": "user", "content": example["question"].strip()},
    #     ],
    #     "completion": [{"role": "assistant", "content": assistant}],
    # }
    return {"prompt": example["question"], "completion": example["answer"]}


def train_model(train_raw, eval_raw, subset, output_dir: str):
    # TRL 用に会話型 prompt-completion へ変換
    cols = train_raw.column_names
    train_ds = train_raw.map(to_conv_prompt_completion, remove_columns=cols)
    eval_ds = eval_raw.map(to_conv_prompt_completion, remove_columns=cols)

    print("First example for training", train_ds[0])

    # 4bit 量子化（QLoRA）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", use_fast=False)
    if tokenizer.pad_token is None:
        print("Pad token has been explicitly set as EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA 設定（Qwen 系の典型的な投影名）
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # SFT 設定
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
        save_steps=25,
        packing=True,
        bf16=True,
        optim="adamw_8bit",
        report_to="wandb",
        run_name=f"qwen3-4b-{subset}" if args.wandb else None,
        completion_only_loss=True,  # prompt は損失から除外（prompt-completion）
        # Qwen3 は tokenizer に chat template が入っているので自動適用される
        # （必要に応じて eos_token を指定可：SFTConfig(eos_token=tokenizer.eos_token)）
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        peft_config=peft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # 学習
    trainer.train()
    trainer.save_model(os.path.join(output_dir, "checkpoint-final"))
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    args = build_args()
    set_seed(args.seed)

    OUTPUT_DIR = os.path.join(args.subset, time.strftime("%m%d%H%M"))

    wandb.init(project="GraphQA-ft")

    # Build system instruction
    # match args.subset:
    #     case "node_count":
    #         TASK_INST = "Answer ONLY with the final number of nodes."
    #     case "edge_count":
    #         TASK_INST = "Answer ONLY with the final number of edges."
    #     case "cycle_check":
    #         TASK_INST = "Answer ONLY with Yes or No."
    #     case _:
    #         raise NotImplementedError(f"Unsupported subset: {args.subset}")

    # SYS_INST = "You are a careful graph reasoning assistant.\n" + TASK_INST
    SYS_INST = ""

    # Load GraphQA dataset
    print(f"[INFO] Load GraphQA: subset={args.subset}")
    train_raw = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_train")
    eval_raw = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_validation")

    # Fine-tune the model using QLoRA
    print("[INFO] Start training")
    train_model(train_raw, eval_raw, args.subset, OUTPUT_DIR)

    # Evaluate the trained model
    if args.do_eval:
        print("[INFO] Start evaluation")
        model_path = os.path.join(OUTPUT_DIR, "checkpoint-final")
        start_time = time.time()
        acc, unknowns = eval_model(model_path, eval_raw, args.subset, SYS_INST)
        wandb.log({"test_accuracy": acc, "test_unknown": unknowns})
        print(f"[INFO] Evaluation completed in {time.time() - start_time:.2f} seconds")
