import argparse
import json
import os
import re
import sys
import time
from typing import List, Literal, Sequence

import torch
import torch.distributed as dist
from datasets import arrow_dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ckpt import _resolve_ckpt_path  # noqa: E402


def is_main_process() -> bool:
    # RANK = 0 is the main process
    return int(os.environ.get("RANK", "0")) == 0


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
    p.add_argument("--loader-workers", type=int, default=0, help="Number of DataLoader worker processes.")
    p.add_argument("--local_rank", type=int, default=None, help=argparse.SUPPRESS)

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


def _distributed_context() -> tuple[int, int, bool]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), True
    return 0, 1, False


def _maybe_init_distributed(local_rank: int | None) -> int:
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if dist.is_available() and world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return local_rank


class _ShardedSequentialSampler(Sampler[int]):
    """Evenly shard indices across distributed ranks without duplication."""

    def __init__(self, dataset_size: int, num_replicas: int, rank: int) -> None:
        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.indices = list(range(rank, dataset_size, num_replicas))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _gather_lists(payload: Sequence[list[str]]) -> list[Sequence[list[str]]]:
    rank, world_size, dist_enabled = _distributed_context()
    if not dist_enabled or world_size == 1:
        return [payload]

    gather_list: list[Sequence[list[str]]] = [None for _ in range(world_size)]  # type: ignore[assignment]
    dist.all_gather_object(gather_list, payload)
    return gather_list


def eval_model(
    model_path: str,
    test_ds: arrow_dataset.Dataset,
    subset: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 64,
    num_workers: int = 0,
    device: torch.device | None = None,
):
    rank, world_size, dist_enabled = _distributed_context()
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None
    if device is None:
        device = (
            torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        )
    model_kwargs = dict(trust_remote_code=True)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.to(device)
    model.eval()
    gen_cfg = GenerationConfig(
        max_new_tokens=8,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    sampler = None
    if world_size > 1:
        sampler = _ShardedSequentialSampler(len(test_ds), world_size, rank)

    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False if sampler is None else False,
        sampler=sampler,
        num_workers=max(0, num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_eval_batch,
    )

    inputs_local, preds_local, refs_local = [], [], []
    progress = tqdm(loader, desc="Evaluating", disable=(rank != 0))

    with torch.inference_mode():
        for batch in progress:
            tokenized = tokenizer(batch["prompt"], return_tensors="pt", padding=True, truncation=True).to(device)
            out = model.generate(**tokenized, generation_config=gen_cfg)

            gens = [gen.split("\nA: ")[-1] for gen in tokenizer.batch_decode(out, skip_special_tokens=True)]
            inputs_local.extend(batch["question"])
            preds_local.extend(gens)
            refs_local.extend(batch["answer"])

    gathered = _gather_lists((inputs_local, preds_local, refs_local))

    if rank == 0:
        inputs, preds, refs = [], [], []
        for chunk_inputs, chunk_preds, chunk_refs in gathered:
            inputs.extend(chunk_inputs)
            preds.extend(chunk_preds)
            refs.extend(chunk_refs)

        acc, unknowns = comp_accuracy(preds, refs, subset)
        print(f"[RESULT] Accuracy (n={len(test_ds)}): {acc:.3f}")
        if unknowns > 0:
            print(f"[RESULT] Unknown Predictions (n={len(test_ds)}): {unknowns}")

        save_dir = "results"
        filename = os.path.join(save_dir, f"{subset}.json")
        os.makedirs(save_dir, exist_ok=True)
        examples = [
            {"question": q, "prediction": p, "ground_truth": r} for q, p, r in list(zip(inputs, preds, refs))[:10]
        ]
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)

        if dist_enabled:
            dist.barrier()
        return acc, unknowns

    if dist_enabled:
        dist.barrier()
    return None, None


if __name__ == "__main__":
    args = build_args()

    if is_main_process():
        print("-" * 16)

    local_rank = _maybe_init_distributed(args.local_rank)

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
        if is_main_process():
            print(f"[INFO] Subset: {args.subset}")
            print(f"[INFO] Number of trials: {args.num_trials}")

    # Load the model
    if args.use_pretrained:
        ckpt_path = args.base_model
        model_type = "pre-trained"
    elif args.model_path is not None:
        ckpt_path, run_name = _resolve_ckpt_path(args.model_path)
        model_type = "fine-tuned"
    else:
        raise ValueError("Either --use-pretrained or --model-path must be specified.")
    if is_main_process():
        print(f"[INFO] Model: {ckpt_path} ({model_type})")

    # Evaluate the model
    start_time = time.time()
    eval_model(
        ckpt_path,
        test_ds,
        args.subset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.loader_workers,
        device=torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu"),
    )
    if is_main_process():
        print(f"[INFO] Evaluation completed in {time.time() - start_time:.2f} seconds")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
