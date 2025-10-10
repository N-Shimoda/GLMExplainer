import argparse
import json
import os
import re
import sys
import time
from typing import List, Literal

import torch
from datasets import arrow_dataset, concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from eval import _resolve_ckpt_path  # noqa: E402


def build_args():
    """
    Parses and returns command-line arguments for fine-tuning a Qwen3-4B model on the GraphQA dataset.

    Returns
    -------
    argparse.Namespace
        An object containing all the parsed command-line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--subset", choices=["node_count", "edge_count", "cycle_check", "triangle_counting"], type=str, required=True
    )
    p.add_argument("--model-path", type=str, help="Checkpoint path of the fine-tuned model.")
    p.add_argument("--use-pretrained", action="store_true", help="Use the pre-trained model without fine-tuning.")
    p.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-4B-Base",
        help="Base model identifier to use when loading the tokenizer or running a pre-trained model.",
    )
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation.")

    # Evaluation settings
    p.add_argument("--num-trials", type=int, default=1, help="Number of trials to run for evaluation.")
    p.add_argument("--quick", action="store_true", help="Run evaluation on a smaller subset for quick testing.")

    return p.parse_args()


def _normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\.$", "", s)
    return s.lower()


def comp_accuracy(
    preds: List[str],
    refs: List[str],
    subset: Literal["cycle_check", "node_count", "edge_count", "triangle_counting", "maximum_flow"],
    exact_match: bool = False,
) -> tuple[float, int]:
    """
    Compute the accuracy of the model's predictions depending on the subset.

    Returns
    -------
    acc : float
        The accuracy of the model's predictions.
    unknowns : int
        The number of unknown predictions.
        This value is only defined for the "cycle_check" subset.
    """
    if subset not in ["cycle_check", "node_count", "edge_count", "triangle_counting", "maximum_flow"]:
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
        case "node_count" | "edge_count" | "triangle_counting" | "maximum_flow":
            digit_ans_li = [ref.strip().split(".")[0] for ref in refs]
            preds = [pred.split("assistant\n")[-1] for pred in preds]
            acc = sum([d in pred for d, pred in zip(digit_ans_li, preds)]) / max(1, len(refs))
            num_unknown = 0

    return acc, num_unknown


def eval_model(model_path, eval_raw: arrow_dataset.Dataset, subset: str, base_model: str, batch_size: int):
    """Evaluate the model on the given dataset.

    Parameters
    ----------
    model_path : str
        Path to the fine-tuned model
    eval_raw : arrow_dataset.Dataset
        The evaluation dataset
    subset : str
        The subset of the GraphQA dataset
    base_model : str
        The base model identifier for loading the tokenizer
    batch_size : int
        Batch size for evaluation

    Returns
    -------
    acc : float
        The accuracy of the model's predictions.
    unknowns : int
        The number of unknown predictions.
    """
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left", use_fast=False, trust_remote_code=True)
    gen_cfg = GenerationConfig(
        max_new_tokens=16,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    inputs, preds, refs = [], [], []
    for batch_start in tqdm(range(0, len(eval_raw), batch_size), "Evaluating"):
        batch = eval_raw[batch_start : batch_start + batch_size]
        user_msgs = [f"{q.strip()}" for q in batch["question"]]
        prompt_strs = [
            tokenizer.apply_chat_template(
                [
                    # {"role": "system", "content": "You are a careful graph reasoner."},
                    {"role": "user", "content": user_msg},
                ],
                tokenize=False,
                add_special_tokens=False,
                continue_final_message=True,
            )
            for user_msg in user_msgs
        ]
        input_ids = tokenizer(prompt_strs, return_tensors="pt", padding=True, truncation=True).to(model.device)
        # input_ids = tokenizer(user_msgs, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(**input_ids, generation_config=gen_cfg)

        gens = [gen.split("\nA: ")[-1] for gen in tokenizer.batch_decode(out, skip_special_tokens=True)]
        inputs.extend(batch["question"])
        preds.extend(gens)
        refs.extend(batch["answer"])

    acc, unknowns = comp_accuracy(preds, refs, subset)
    print(f"[RESULT] Accuracy (n={len(eval_raw)}): {acc:.3f}")
    if unknowns > 0:
        print(f"[RESULT] Unknown Predictions (n={len(eval_raw)}): {unknowns}")

    # Save 10 examples to JSON
    save_dir = "results"
    filename = os.path.join(save_dir, f"{subset}.json")
    os.makedirs(save_dir, exist_ok=True)
    examples = [{"question": q, "prediction": p, "ground_truth": r} for q, p, r in list(zip(inputs, preds, refs))[:10]]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    return acc, unknowns


if __name__ == "__main__":
    args = build_args()
    print("-" * 12)

    # Dataset
    if args.quick:
        if args.num_trials > 1:
            print("[WARNING] --quick is enabled; num_trials will be set to 1.")
        N = 96
        test_ds = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_test")
        test_ds = test_ds.select(range(N))  # for quick testing
        print(f"Subset: {args.subset} (top {N} samples)")
    else:
        test_ds = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_test")
        test_ds = concatenate_datasets([test_ds] * args.num_trials)
        print(f"Subset: {args.subset}")
        print(f"Number of trials: {args.num_trials}")

    # Load the model
    if args.use_pretrained:
        ckpt_path = args.base_model
        print(f"Model: {ckpt_path} (pre-trained)")
    elif args.model_path is not None:
        ckpt_path, run_name = _resolve_ckpt_path(args.model_path)
        print(f"Model: {ckpt_path} (fine-tuned)")
    else:
        raise ValueError("Either --use-pretrained or --model-path must be specified.")

    # Evaluate the model
    start_time = time.time()
    eval_model(ckpt_path, test_ds, args.subset, args.base_model, args.batch_size)
    print(f"[INFO] Evaluation completed in {time.time() - start_time:.2f} seconds")
