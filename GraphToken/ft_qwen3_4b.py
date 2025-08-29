"""
Qwen/Qwen3-4B-Instruct-2507 を GraphQA で QLoRA (4bit) 微調整
- データ: baharef/GraphQA から subset/split を指定
- 方式: TRL SFTTrainer + PEFT(LoRA) + bitsandbytes 4bit (QLoRA)
- 評価: 簡易 Exact Match（空白/改行/末尾ピリオド無視）
"""

import argparse
import json
import os
import time
from typing import Dict, List, Literal

import torch
from accelerate.utils import set_seed
from datasets import arrow_dataset, load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from trl import SFTConfig, SFTTrainer
from utils import _normalize_text


def build_args():
    """
    Parses and returns command-line arguments for fine-tuning a Qwen3-4B model on the GraphQA dataset.

    Returns
    -------
    argparse.Namespace
        An object containing all the parsed command-line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="base model")
    p.add_argument(
        "--subset",
        type=str,
        default="cycle_check",
        help="GraphQA subset（see https://huggingface.co/datasets/baharef/GraphQA）",
    )
    p.add_argument("--output_dir", type=str, default=None)  # "qwen3-4b-graphqa-qlora"
    p.add_argument("--wandb", action="store_true", help="Use Weights & Biases for logging")
    p.add_argument("--seed", type=int, default=42)

    # Execution flow
    p.add_argument("--do_train", action="store_true", help="Train the model")
    p.add_argument("--do_eval", action="store_true", help="Evaluate the model")

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
    assistant = example["answer"].strip()
    return {
        "prompt": [
            {"role": "system", "content": SYS_INST},
            {"role": "user", "content": example["question"].strip()},
        ],
        "completion": [{"role": "assistant", "content": assistant}],
    }


def comp_accuracy(
    preds: List[str],
    refs: List[str],
    subset: Literal["cycle_check", "node_count", "edge_count"],
    exact_match: bool = False,
) -> float:
    if subset not in ["cycle_check", "node_count", "edge_count"]:
        raise NotImplementedError(f"Unsupported subset: {subset}")

    match subset:
        case "cycle_check":
            if exact_match:
                acc = sum(_normalize_text(p) == _normalize_text(r) for p, r in zip(preds, refs)) / max(1, len(refs))
                num_unknown = 0
            else:
                low_preds = [pred.lower() for pred in preds]
                low_refs = [ref.lower() for ref in refs]
                preds_yes_no = ["yes" if "yes" in pred else "no" if "no" in pred else "unknown" for pred in low_preds]
                refs_yes_no = ["yes" if "yes" in ref else "no" if "no" in ref else "unknown" for ref in low_refs]
                acc = sum(p == r for p, r in zip(preds_yes_no, refs_yes_no)) / max(1, len(refs_yes_no))
                num_unknown = sum(p == "unknown" for p in preds_yes_no)
        case "node_count" | "edge_count":
            digit_ans_li = [ref.strip().split(".")[0] for ref in refs]
            preds = [pred.split("assistant\n")[-1] for pred in preds]
            print("preds", preds)
            print("digit_ans_li", digit_ans_li)
            acc = sum([d in pred for d, pred in zip(digit_ans_li, preds)]) / max(1, len(refs))
            num_unknown = 0

    return acc, num_unknown


def eval_model(
    model_path, eval_raw: arrow_dataset.Dataset, subset: Literal["cycle_check", "node_count", "edge_count"]
):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", padding_side="left", use_fast=False)
    gen_cfg = GenerationConfig(
        max_new_tokens=32,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

    inputs, preds, refs = [], [], []
    batch_size = 16
    for batch_start in tqdm(range(0, len(eval_raw), batch_size), "Evaluating"):
        batch = eval_raw[batch_start : batch_start + batch_size]
        user_msgs = [f"{q.strip()}" for q in batch["question"]]
        prompt_strs = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYS_INST},
                    {"role": "user", "content": user_msg},
                ],
                tokenize=False,
                # continue_final_message=True,
                add_generation_prompt=True,
            )
            for user_msg in user_msgs
        ]

        input_ids = tokenizer(prompt_strs, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(**input_ids, generation_config=gen_cfg)

        gens = tokenizer.batch_decode(out, skip_special_tokens=True)
        inputs.extend(batch["question"])
        preds.extend(gens)
        refs.extend(batch["answer"])

    acc, unknowns = comp_accuracy(preds, refs, subset)
    print(f"[RESULT] Accuracy (n={len(eval_raw)}): {acc:.3f}")
    if unknowns > 0:
        print(f"[RESULT] Unknown Predictions (n={len(eval_raw)}): {unknowns}")

    # Save 10 examples to JSON
    examples = [{"question": q, "prediction": p, "ground_truth": r} for q, p, r in list(zip(inputs, preds, refs))[:10]]
    filename = f"examples-{subset}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved 10 examples to {filename}")


def train_model(train_raw, eval_raw, subset):
    # TRL 用に会話型 prompt-completion へ変換
    cols = train_raw.column_names
    train_ds = train_raw.map(to_conv_prompt_completion, remove_columns=cols)
    eval_ds = eval_raw.map(to_conv_prompt_completion, remove_columns=cols)

    print("First sample", train_ds[0])

    # 4bit 量子化（QLoRA）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        print("Here!")
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
        output_dir=args.output_dir,
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
        report_to="wandb" if args.wandb else "none",
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
    trainer.save_model(os.path.join(args.output_dir, "checkpoint-final"))
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    args = build_args()
    set_seed(args.seed)

    # Build system instruction
    match args.subset:
        case "node_count":
            TASK_INST = "Answer ONLY with the final number of nodes."
        case "edge_count":
            TASK_INST = "Answer ONLY with the final number of edges."
        case "cycle_check":
            TASK_INST = "Answer ONLY with Yes or No."
        case _:
            raise NotImplementedError(f"Unsupported subset: {args.subset}")

    SYS_INST = "You are a careful graph reasoning assistant.\n" + TASK_INST

    # Load GraphQA dataset
    print(f"[INFO] Load GraphQA: subset={args.subset}")
    train_raw = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_train")
    eval_raw = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_validation")

    if args.do_train:
        print("[INFO] Start training")
        train_model(train_raw, eval_raw, args.subset)

    if args.do_eval:
        if args.output_dir:
            model_path = os.path.join(args.output_dir, "checkpoint-final")
            print(f"[INFO] Evaluation on a fine-tuned model: {model_path}")
        else:
            model_path = args.model_name
            print(f"[INFO] Evaluation on a pre-trained model: {model_path}")

        start_time = time.time()
        eval_model(model_path, eval_raw, args.subset)
        print(f"[INFO] Evaluation completed in {time.time() - start_time:.2f} seconds")
        print(f"[INFO] Evaluation completed in {time.time() - start_time:.2f} seconds")
