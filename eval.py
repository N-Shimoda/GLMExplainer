import argparse
import math
import os
from typing import List, Literal, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.collator import GraphQACollator
from src.glm import GraphTokenLM
from src.preprocess import add_graph_column


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--subset",
        type=str,
        choices=["node_count", "edge_count", "cycle_check", "triangle_counting", "maximum_flow"],
        default="edge_count",
    )
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--num_graph_tokens", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=32)
    return p.parse_args()


def _checkpoint_step(path: str) -> int:
    name = os.path.basename(path.rstrip(os.sep))
    try:
        return int(name.split("-")[-1])
    except (ValueError, IndexError):
        return -1


def _resolve_checkpoint_path(model_path: str) -> str:
    """Resolve the concrete checkpoint directory to load."""
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        if os.path.isfile(config_path):
            return model_path

        candidates = [
            os.path.join(model_path, entry)
            for entry in os.listdir(model_path)
            if entry.startswith("checkpoint-") and os.path.isdir(os.path.join(model_path, entry))
        ]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint-* directories found under '{model_path}'.")

        candidates.sort(key=lambda p: (_checkpoint_step(p), p))
        best = candidates[-1]
        if _checkpoint_step(best) < 0:
            raise FileNotFoundError(
                f"Could not infer the last checkpoint under '{model_path}'. Provide a direct checkpoint path."
            )
        return best

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint path '{model_path}' does not exist.")

    return model_path


def _normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("\n", " ").replace("\t", " ")
    s = "".join(s.split())
    if s.endswith("."):
        s = s[:-1]
    return s.lower()


def comp_accuracy(
    preds: List[str],
    refs: List[str],
    subset: Literal["cycle_check", "node_count", "edge_count", "triangle_counting", "maximum_flow"],
    exact_match: bool = False,
) -> Tuple[float, int]:
    if subset not in {"cycle_check", "node_count", "edge_count", "triangle_counting", "maximum_flow"}:
        raise NotImplementedError(f"Unsupported subset: {subset}")

    if subset == "cycle_check":
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
    else:
        digit_refs = [ref.strip().split(".")[0] for ref in refs]
        filtered_preds = [pred.split("assistant\n")[-1] for pred in preds]
        acc = sum(d in pred for d, pred in zip(digit_refs, filtered_preds)) / max(1, len(refs))
        num_unknown = 0

    return acc, num_unknown


def eval_model(
    model_path: str,
    eval_dataset,
    num_graph_tokens: int,
    *,
    batch_size: int = 4,
    subset: str,
    max_new_tokens: int,
):
    if "answer" not in eval_dataset.column_names:
        raise KeyError("Evaluation dataset does not contain 'answer' column required for accuracy computation.")

    checkpoint_path = _resolve_checkpoint_path(model_path)
    print(f"Loading checkpoint from: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    glm = GraphTokenLM.from_pretrained(checkpoint_path)
    glm.to(device)
    glm.eval()

    tokenizer = AutoTokenizer.from_pretrained(glm.config.llm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_field = "task_description" if "task_description" in eval_dataset.column_names else "question"
    if text_field not in eval_dataset.column_names:
        raise KeyError(f"'{text_field}' not found in evaluation dataset columns: {eval_dataset.column_names}")

    base_collator = GraphQACollator(
        tokenizer=tokenizer,
        text_field=text_field,
        max_length=512,
        num_graph_tokens=num_graph_tokens,
    )

    def collate_fn(features):
        batch = base_collator(features)
        batch["prompts"] = [f[text_field] for f in features]
        batch["answers"] = [f.get("answer", "") for f in features]
        return batch

    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    loss_sum = 0.0
    total_tokens = 0
    num_batches = 0
    preds: List[str] = []
    refs: List[str] = []

    progress = tqdm(
        dataloader,
        desc="Evaluating",
        total=max(1, math.ceil(len(eval_dataset) / batch_size)),
    )

    for batch in progress:
        num_batches += 1
        graph_batch = batch["graph"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = glm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                graph=graph_batch,
            )

        batch_loss = outputs.loss.detach()
        valid_tokens = (labels != -100).sum().item()
        if valid_tokens > 0:
            loss_sum += batch_loss.item() * valid_tokens
            total_tokens += valid_tokens

        with torch.no_grad():
            gen_sequences = glm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph=graph_batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_sequences = gen_sequences.cpu()
        prompt_lengths = (batch["input_ids"] != tokenizer.pad_token_id).sum(dim=1).tolist()

        for seq, prompt_len in zip(gen_sequences, prompt_lengths):
            gen_tokens = seq[int(prompt_len) :]
            preds.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())

        refs.extend(batch["answers"])

        progress.set_postfix({"loss": f"{batch_loss.item():.4f}"})

    metrics = {"num_batches": num_batches, "num_tokens": total_tokens, "num_examples": len(eval_dataset)}
    if total_tokens == 0:
        metrics["eval_loss"] = float("nan")
        metrics["perplexity"] = float("nan")
        print("No valid tokens found during evaluation.")
    else:
        avg_loss = loss_sum / total_tokens
        metrics["eval_loss"] = avg_loss
        try:
            metrics["perplexity"] = math.exp(avg_loss)
        except OverflowError:
            metrics["perplexity"] = float("inf")

    accuracy, num_unknown = comp_accuracy(preds, refs, subset=subset)
    metrics["accuracy"] = accuracy
    metrics["num_unknown"] = num_unknown

    if total_tokens > 0:
        ppl = metrics["perplexity"]
        ppl_str = f"{ppl:.4f}" if isinstance(ppl, (int, float)) and math.isfinite(ppl) else "inf"
        print(
            f"[Eval] loss={metrics['eval_loss']:.4f} | perplexity={ppl_str} | "
            f"tokens={total_tokens} | batches={num_batches}"
        )

    print(f"[Eval] accuracy={accuracy:.4f} (n={len(refs)})")
    if num_unknown > 0:
        print(f"[Eval] unknown predictions={num_unknown}")

    return metrics


if __name__ == "__main__":
    args = build_args()

    train_ds = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_train")
    eval_ds = load_dataset(
        "baharef/GraphQA",
        args.subset,
        split="zero_shot_validation" if args.subset != "maximum_flow" else "zero_shot_test",
    )

    train_ds = train_ds.map(add_graph_column, desc="add_graph_column(train)")
    eval_ds = eval_ds.map(add_graph_column, desc="add_graph_column(eval)")

    eval_model(
        args.model_path,
        eval_ds,
        args.num_graph_tokens,
        batch_size=args.batch_size,
        subset=args.subset,
        max_new_tokens=args.max_new_tokens,
    )
