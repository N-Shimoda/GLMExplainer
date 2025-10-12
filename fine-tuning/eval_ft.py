import argparse
import importlib.util
import json
import os
import re
import sys
import time
from typing import List, Literal

import torch
from datasets import arrow_dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

_EVAL_MODULE_NAME = "graph_token_repo_eval"
_EVAL_MODULE_PATH = os.path.join(ROOT_DIR, "eval_ft.py")

if _EVAL_MODULE_NAME in sys.modules:
    _eval_module = sys.modules[_EVAL_MODULE_NAME]
else:
    spec = importlib.util.spec_from_file_location(_EVAL_MODULE_NAME, _EVAL_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load spec for eval module at '{_EVAL_MODULE_PATH}'.")
    _eval_module = importlib.util.module_from_spec(spec)
    sys.modules[_EVAL_MODULE_NAME] = _eval_module
    spec.loader.exec_module(_eval_module)

if not hasattr(_eval_module, "_resolve_ckpt_path"):
    raise ImportError(f"Module loaded from '{_EVAL_MODULE_PATH}' missing '_resolve_ckpt_path'.")

_resolve_ckpt_path = _eval_module._resolve_ckpt_path


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
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation.")
    p.add_argument("--loader-workers", type=int, default=0, help="Number of DataLoader worker processes.")

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


def _with_prompts(dataset: arrow_dataset.Dataset, tokenizer: PreTrainedTokenizerBase) -> arrow_dataset.Dataset:
    def _build_prompts(batch: dict[str, list[str]]) -> dict[str, list[str]]:
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": question.strip()},
                ],
                tokenize=False,
                add_special_tokens=False,
                continue_final_message=True,
            )
            for question in batch["question"]
        ]
        return {"prompt": prompts}

    dataset = dataset.map(_build_prompts, batched=True, desc="Preparing prompts")
    return dataset.select_columns([col for col in dataset.column_names if col in {"prompt", "question", "answer"}])


def _collate_eval_batch(batch: list[dict[str, str]]) -> dict[str, list[str]]:
    return {
        "prompt": [row["prompt"] for row in batch],
        "question": [row["question"] for row in batch],
        "answer": [row["answer"] for row in batch],
    }


def eval_model(
    model_path: str,
    eval_raw: arrow_dataset.Dataset,
    subset: str,
    *,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    num_workers: int,
):
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None
    model_kwargs = dict(device_map="auto", trust_remote_code=True)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    gen_cfg = GenerationConfig(
        max_new_tokens=8,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    inputs, preds, refs = [], [], []
    loader = DataLoader(
        eval_raw,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_eval_batch,
    )

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating"):
            tokenized = tokenizer(batch["prompt"], return_tensors="pt", padding=True, truncation=True).to(model.device)
            out = model.generate(**tokenized, generation_config=gen_cfg)

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

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="left",
        use_fast=False,
        trust_remote_code=True,
    )

    # Dataset
    if args.quick:
        if args.num_trials > 1:
            print("[WARNING] --quick is enabled; num_trials will be set to 1.")
        N = 96
        test_raw = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_test")
        test_raw = test_raw.select(range(N))  # for quick testing
        test_ds = _with_prompts(test_raw, tokenizer)
        print(f"Subset: {args.subset} (top {N} samples)")
    else:
        test_raw = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_test")
        test_ds = _with_prompts(test_raw, tokenizer)
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
    eval_model(
        ckpt_path,
        test_ds,
        args.subset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.loader_workers,
    )
    print(f"[INFO] Evaluation completed in {time.time() - start_time:.2f} seconds")
