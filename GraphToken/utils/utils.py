import json
import os
import re
from typing import List, Literal

import torch
from datasets import arrow_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def _normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\.$", "", s)
    return s.lower()


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
    model_path,
    eval_raw: arrow_dataset.Dataset,
    subset: Literal["cycle_check", "node_count", "edge_count"],
    SYS_INST: str,
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
    save_dir = "examples"
    filename = os.path.join(save_dir, f"{subset}.json")
    os.makedirs(save_dir, exist_ok=True)
    examples = [{"question": q, "prediction": p, "ground_truth": r} for q, p, r in list(zip(inputs, preds, refs))[:10]]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved 10 examples to {filename}")

    return acc, unknowns
