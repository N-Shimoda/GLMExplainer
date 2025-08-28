"""
Qwen/Qwen3-4B-Instruct-2507 を GraphQA で QLoRA (4bit) 微調整
- データ: baharef/GraphQA から subset/split を指定
- 方式: TRL SFTTrainer + PEFT(LoRA) + bitsandbytes 4bit (QLoRA)
- 評価: 簡易 Exact Match（空白/改行/末尾ピリオド無視）
"""

import argparse
import os
import re
from typing import Dict, List

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


def build_args():
    """
    Parses and returns command-line arguments for fine-tuning a Qwen3-4B model on the GraphQA dataset.

    Returns
    -------
    argparse.Namespace
        An object containing all the parsed command-line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="ベースモデル")
    p.add_argument(
        "--subset",
        type=str,
        default="cycle_check",
        help="GraphQA subset（see https://huggingface.co/datasets/baharef/GraphQA）",
    )
    p.add_argument("--train_split", type=str, default="zero_shot_train")
    p.add_argument("--eval_split", type=str, default="zero_shot_validation")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--wandb", action="store_true", help="Use Weights & Biases for logging")
    p.add_argument("--seed", type=int, default=42)

    # Execution flow
    p.add_argument("--do_train", action="store_true", help="Train the model")
    p.add_argument("--do_eval", action="store_true", help="Evaluate the model")

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


def _normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\.$", "", s)
    return s.lower()


def count_corrects(preds: List[str], refs: List[str], exact_match: bool = False) -> float:
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
    return acc, num_unknown


def eval_model(model_path, eval_raw: arrow_dataset.Dataset):
    # eval_raw = eval_raw.select(range(48))  # dev

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", padding_side="left", use_fast=False)
    gen_cfg = GenerationConfig(
        max_new_tokens=16,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

    inputs, preds, refs = [], [], []
    batch_size = 16
    for batch_start in tqdm(range(0, len(eval_raw), batch_size), "Evaluating"):
        batch = eval_raw[batch_start : batch_start + batch_size]
        # user_msgs = [f"{SYS_INST}\n\n{q.strip()}" for q in batch["question"]]
        user_msgs = [f"{q.strip()}" for q in batch["question"]]
        prompt_strs = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYS_INST},
                    {"role": "user", "content": user_msg},
                    # {"role": "assistant", "content": "A: "},
                ],
                tokenize=False,
                continue_final_message=True,
                # add_generation_prompt=True,
            )
            for user_msg in user_msgs
        ]

        input_ids = tokenizer(prompt_strs, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(**input_ids, generation_config=gen_cfg)

        # prompt_lens = input_ids["attention_mask"].sum(dim=1).tolist()
        # gens = [
        #     tokenizer.decode(out[i][prompt_lens[i] :], skip_special_tokens=True).strip() for i in range(out.size(0))
        # ]
        gens = tokenizer.batch_decode(out, skip_special_tokens=True)
        inputs.extend(batch["question"])
        preds.extend(gens)
        refs.extend(batch["answer"])

    acc, unknowns = count_corrects(preds, refs)
    print(f"[RESULT] Yes/No Accuracy (n={len(eval_raw)}): {acc:.3f}")
    if unknowns > 0:
        print(f"[RESULT] Unknown Predictions (n={len(eval_raw)}): {unknowns}")

    # show 3 examples
    for q, p, r in list(zip(inputs, preds, refs))[:3]:
        print(f"Question: {q}")
        print(f"Prediction: {p}")
        print(f"Ground Truth: {r}")
        print("---")


def train_model(train_ds, eval_ds):
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
        save_total_limit=2,
        packing=True,
        bf16=True,
        optim="adamw_8bit",
        report_to="wandb" if args.wandb else "none",
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


def main(args):

    # Load GraphQA dataset
    print(f"[INFO] Load GraphQA: subset={args.subset}, train={args.train_split}, eval={args.eval_split}")
    train_raw = load_dataset("baharef/GraphQA", args.subset, split=args.train_split)
    eval_raw = load_dataset("baharef/GraphQA", args.subset, split=args.eval_split)

    # TRL 用に会話型 prompt-completion へ変換
    cols = train_raw.column_names
    train_ds = train_raw.map(to_conv_prompt_completion, remove_columns=cols)
    eval_ds = eval_raw.map(to_conv_prompt_completion, remove_columns=cols)

    if args.do_train:
        print("[INFO] Start training")
        train_model(train_ds, eval_ds)

    if args.do_eval and args.output_dir:
        model_path = os.path.join(args.output_dir, "checkpoint-final")
        print("[INFO] Start evaluation on a fine-tuned model")
        print(f"[INFO] Model path: {model_path}")
        eval_model(model_path, eval_raw)
    elif args.do_eval:
        print("[INFO] Start evaluation on a pre-trained model")
        eval_model(args.model_name, eval_raw)


if __name__ == "__main__":
    args = build_args()
    set_seed(args.seed)
    main(args)
