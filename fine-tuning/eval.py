import argparse
import json
import os
import re
import time
from typing import List, Literal

import torch
from datasets import arrow_dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


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
        "--subset",
        choices=["node_count", "edge_count", "cycle_check", "triangle_counting", "maximum_flow"],
        type=str,
        required=True,
        help="Specifies GraphQA subset（https://huggingface.co/datasets/baharef/GraphQA）",
    )
    p.add_argument("--model_path", type=str, default=None, help="Checkpoint path of the fine-tuned model.")
    p.add_argument("--quick", action="store_true", help="Whether to run in quick mode.")

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


def eval_model(model_path, eval_raw: arrow_dataset.Dataset, subset: str):
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

        gens = tokenizer.batch_decode(out, skip_special_tokens=True)
        inputs.extend(batch["question"])
        preds.extend(gens)
        refs.extend(batch["answer"])

    acc, unknowns = comp_accuracy(preds, refs, subset)
    print(f"[RESULT] Accuracy (n={len(eval_raw)}): {acc:.3f}")
    if unknowns > 0:
        print(f"[RESULT] Unknown Predictions (n={len(eval_raw)}): {unknowns}")

    # Save 10 examples to JSON
    save_dir = "examples"
    filename = os.path.join(save_dir, f"{subset}.json")
    os.makedirs(save_dir, exist_ok=True)
    examples = [{"question": q, "prediction": p, "ground_truth": r} for q, p, r in list(zip(inputs, preds, refs))[:10]]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    return acc, unknowns


if __name__ == "__main__":

    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    args = build_args()
    print("-" * 12)

    # Dataset
    if args.quick:
        N = 96
        test_ds = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_test")
        test_ds = test_ds.select(range(N))  # for quick testing
        print(f"Subset: {args.subset} (top {N} samples)")
    else:
        test_ds = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_test")
        print(f"Subset: {args.subset}")

    # Load the model
    if args.model_path:
        model_path = args.model_path
        print(f"Model: {model_path} (fine-tuned)")
    else:
        model_path = MODEL_NAME
        print(f"Model: {model_path} (pre-trained)")

    # Evaluate the model
    start_time = time.time()
    eval_model(model_path, test_ds, args.subset)
    print(f"[INFO] Evaluation completed in {time.time() - start_time:.2f} seconds")
