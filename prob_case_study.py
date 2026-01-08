import argparse
import math

import torch
from transformers import AutoTokenizer

from eval import create_pyg_batch
from src.ckpt import _resolve_ckpt_path
from src.constants import MOTIFQA_SUBSETS
from src.explanation.preprocess import build_dataset, filter_dataset
from src.glm import GraphTokenLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--subset", type=str, required=True, choices=MOTIFQA_SUBSETS)
    p.add_argument("--sample-idx", type=int, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    # Load model and tokenizer
    ckpt_path, run_name = _resolve_ckpt_path(args.model_path)
    model = GraphTokenLM.from_pretrained(ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tok = AutoTokenizer.from_pretrained(model.config.llm_name, trust_remote_code=True)
    print("Loaded model from {}".format(ckpt_path))

    # Prepare dataset sample
    dataset = build_dataset("MotifQA", args.subset, "test", node_feat_dim=model.config.node_feat_dim)
    dataset = filter_dataset(dataset, "MotifQA", sample_idx=args.sample_idx)
    sample = dataset[0]
    prompt = sample["prompt"]
    completion = sample["completion"]
    graph = create_pyg_batch(sample["graph"], device)

    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    completion_ids = tok.encode(completion, add_special_tokens=False)
    if not completion_ids:
        raise ValueError("Completion encodes to zero tokens. Provide a non-empty completion.")

    input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, graph=graph)

    log_probs = torch.log_softmax(outputs.logits, dim=-1)[0]
    num_graph_tokens = int(model.config.num_graph_tokens)
    base_pos = num_graph_tokens + len(prompt_ids) - 1
    if base_pos < 0:
        raise ValueError("Prompt is empty and graph tokens are disabled; cannot score completion tokens.")

    token_rows = []
    for idx, token_id in enumerate(completion_ids):
        pos = base_pos + idx
        if pos >= log_probs.size(0):
            raise ValueError("Token position exceeds model logits length.")
        log_prob = log_probs[pos, token_id].item()
        token_rows.append((token_id, log_prob, math.exp(log_prob)))

    print("Prompt:", prompt)
    print("Completion:", completion)
    print("Completion token probabilities:")
    for token_id, log_prob, prob in token_rows:
        token_str = tok.convert_ids_to_tokens([token_id])[0]
        print(f"{token_str}\t(id={token_id})\tprob={prob:.6g}\tlog_prob={log_prob:.6g}")
    total_log_prob = sum(lp for _, lp, _ in token_rows)
    mean_log_prob = total_log_prob / len(token_rows)
    print(f"Total log_prob: {total_log_prob:.6g}")
    print(f"Mean log_prob/token: {mean_log_prob:.6g}")


if __name__ == "__main__":
    main()
