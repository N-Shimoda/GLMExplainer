#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen/Qwen3-4B-Instruct-2507 を GraphQA で QLoRA (4bit) 微調整
- データ: baharef/GraphQA から subset/split を指定
- 方式: TRL SFTTrainer + PEFT(LoRA) + bitsandbytes 4bit (QLoRA)
- 評価: 簡易 Exact Match（空白/改行/末尾ピリオド無視）
"""

import argparse
import re
from typing import Dict, List

import torch
from accelerate.utils import set_seed
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from trl import SFTConfig, SFTTrainer


# ---------- 引数 ----------
def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="ベースモデル")
    p.add_argument(
        "--subset",
        type=str,
        default="connected_nodes",
        help="GraphQA subset（see https://huggingface.co/datasets/baharef/GraphQA）",
    )
    p.add_argument("--train_split", type=str, default="zero_shot_train")
    p.add_argument("--eval_split", type=str, default="zero_shot_validation")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)

    # 主要ハイパラ
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


# ---------- データ前処理 ----------
SYS_INST = (
    "You are a careful graph reasoning assistant.\n"
    "Answer ONLY with the final list exactly as in the dataset (e.g., '1, 2, 4' or 'No nodes.')."
)


def to_conv_prompt_completion(example: Dict) -> Dict:
    """
    TRL SFTTrainer が理解する「会話型 prompt-completion」形式に変換
      {
        "prompt":    [{"role": "user", "content": "<指示>"}],
        "completion":[{"role": "assistant", "content": "<解答>"}]
      }
    """
    user = f"{SYS_INST}\n\n{example['question'].strip()}"
    assistant = example["answer"].strip()
    return {
        "prompt": [{"role": "user", "content": user}],
        "completion": [{"role": "assistant", "content": assistant}],
    }


# ---------- 簡易評価（Exact Match） ----------
def _normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\.$", "", s)
    return s.lower()


def exact_match(preds: List[str], refs: List[str]) -> float:
    correct = sum(_normalize_text(p) == _normalize_text(r) for p, r in zip(preds, refs))
    return correct / max(1, len(refs))


# ---------- メイン ----------
def main():
    args = build_args()
    set_seed(args.seed)

    # 4bit 量子化（QLoRA）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"[INFO] Load model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
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

    # GraphQA 読み込み
    print(f"[INFO] Load GraphQA: subset={args.subset}, train={args.train_split}, eval={args.eval_split}")
    train_raw = load_dataset("baharef/GraphQA", args.subset, split=args.train_split)
    eval_raw = load_dataset("baharef/GraphQA", args.subset, split=args.eval_split)

    # TRL 用に会話型 prompt-completion へ変換
    cols = train_raw.column_names
    train_ds = train_raw.map(to_conv_prompt_completion, remove_columns=cols)
    eval_ds = eval_raw.map(to_conv_prompt_completion, remove_columns=cols)

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
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        packing=True,
        bf16=True,
        optim="adamw_8bit",
        report_to="wandb",
        completion_only_loss=True,  # prompt は損失から除外（prompt-completion）
        # Qwen3 は tokenizer に chat template が入っているので自動適用される
        # （必要に応じて eos_token を指定可：SFTConfig(eos_token=tokenizer.eos_token)）
        model_init_kwargs={
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
        },
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
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # ---------- 簡易評価（生成 → EM） ----------
    print("[INFO] Evaluate (greedy, temperature=0)")
    gen_cfg = GenerationConfig(
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
    )

    n_eval = min(20, len(eval_raw))
    preds, refs, inputs = [], [], []
    for i in range(n_eval):
        ex = eval_raw[i]
        user_msg = f"{SYS_INST}\n\n{ex['question'].strip()}"
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        # with torch.no_grad():
        #     out = trainer.model.generate(**input_ids, generation_config=gen_cfg)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = trainer.model.generate(**input_ids, generation_config=gen_cfg)
        gen = tokenizer.decode(out[0][input_ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        preds.append(gen)
        refs.append(ex["answer"])
        inputs.append(ex["question"])

    acc = exact_match(preds, refs)
    print(f"[RESULT] Exact Match (n={n_eval}): {acc:.3f}")
    for q, p, r in list(zip(inputs, preds, refs))[:3]:
        print("Q>", q[:80].replace("\n", " ") + ("..." if len(q) > 80 else ""))
        print("P>", p)
        print("G>", r)
        print("---")


if __name__ == "__main__":
    main()
