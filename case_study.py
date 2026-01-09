import argparse
import math
import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from eval import create_pyg_batch
from src.ckpt import _resolve_ckpt_path
from src.constants import MOTIFQA_SUBSETS
from src.explanation.args import check_non_negative_int
from src.explanation.preprocess import build_dataset, filter_dataset
from src.glm import GraphTokenLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--ckpt-index", type=int, default=-1)
    p.add_argument("--subset", type=str, required=True, choices=MOTIFQA_SUBSETS)
    p.add_argument(
        "--target-pos-samples",
        action="store_true",
        help="If set, only explain positive samples (graphs containing house motifs).",
    )
    p.add_argument(
        "--num-samples", type=check_non_negative_int, default=None, help="Number of samples to explain (default: None)"
    )
    p.add_argument(
        "--sample-idx",
        type=check_non_negative_int,
        default=None,
        help="Specify the index of the sample to explain (default: None)",
    )
    p.add_argument(
        "--baseline-graph",
        type=str,
        default="empty",
        choices=["empty", "complete", "random"],
        help="Type of baseline graph to use.",
    )
    p.add_argument("--output-dir", type=str, default="plots/", help="Directory to save output plots.")
    p.add_argument("--verbose", action="store_true", help="If set, print token probabilities.")
    return p.parse_args()


def plot_prob_comparison(org_token_probs, base_token_probs, output_path: str = "plots/token_prob_comparison.png"):
    if len(org_token_probs) != len(base_token_probs):
        raise ValueError("Token probability lists must be the same length.")

    tokens = []
    org_probs = []
    base_probs = []
    for org_row, base_row in zip(org_token_probs, base_token_probs):
        org_id, org_token, _, org_prob = org_row
        base_id, base_token, _, base_prob = base_row
        if org_id != base_id or org_token != base_token:
            raise ValueError("Token sequences do not match between runs.")
        tokens.append(org_token)
        org_probs.append(org_prob)
        base_probs.append(base_prob)

    xs = list(range(len(tokens)))
    plt.figure(figsize=(10, 4))
    plt.plot(xs, org_probs, marker="o", linewidth=1.5, label="w/ graph")
    plt.plot(xs, base_probs, marker="x", linewidth=1.5, label="w/o graph")
    plt.xticks(xs, tokens, rotation=40, ha="right")
    plt.ylabel("Probability")
    plt.title("Token Probability Comparison")
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def comp_token_probs(model: GraphTokenLM, tok: AutoTokenizer, sample: dict) -> list[tuple[int, str, float, float]]:
    """Compute token probabilities for the completion tokens given the prompt and graph.

    Parameters
    ----------
    model : GraphTokenLM
        Graph-Language Model based on GraphToken architecture.
    tok : AutoTokenizer
        Tokenizer corresponding to the language model.
    sample : dict
        A data sample containing "prompt", "completion", and "graph".

    Returns
    -------
    list[tuple[int, str, float, float]]
        A list of tuples for each completion token: (token_id, token_str, log_prob, prob).
    """
    prompt = sample["prompt"]
    completion = sample["completion"]
    graph = create_pyg_batch(sample["graph"], model.device)

    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    completion_ids = tok.encode(completion, add_special_tokens=False)
    if not completion_ids:
        raise ValueError("Completion encodes to zero tokens. Provide a non-empty completion.")
    input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, graph=graph)

    log_probs = torch.log_softmax(outputs.logits, dim=-1)[0]
    num_graph_tokens = model.config.num_graph_tokens
    base_pos = num_graph_tokens + len(prompt_ids) - 1
    if base_pos < 0:
        raise ValueError("Prompt is empty and graph tokens are disabled; cannot score completion tokens.")

    token_rows = []
    for idx, token_id in enumerate(completion_ids):
        pos = base_pos + idx
        if pos >= log_probs.size(0):
            raise ValueError("Token position exceeds model logits length.")
        log_prob = log_probs[pos, token_id].item()
        token_str = tok.convert_ids_to_tokens([token_id])[0]
        token_rows.append((token_id, token_str, log_prob, math.exp(log_prob)))

    return token_rows


def main():
    args = parse_args()
    OUT_DIR = os.path.join(args.output_dir, args.subset, args.baseline_graph)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load model and tokenizer
    ckpt_path, _ = _resolve_ckpt_path(args.model_path, ckpt_index=args.ckpt_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphTokenLM.from_pretrained(ckpt_path)
    model = model.to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model.config.llm_name, trust_remote_code=True)
    print("Loaded model from {}".format(ckpt_path))

    # Prepare dataset sample
    dataset = build_dataset("MotifQA", args.subset, "test", node_feat_dim=model.config.node_feat_dim)
    dataset = filter_dataset(
        dataset,
        "MotifQA",
        target_pos_samples=args.target_pos_samples,
        sample_idx=args.sample_idx,
        num_samples=args.num_samples,
    )

    for sample in tqdm(dataset, desc="Processing samples"):
        # Original input
        org_token_probs = comp_token_probs(model, tok, sample)

        # Alternated input
        num_nodes = len(set(sample["nodes"]))
        match args.baseline_graph:
            case "empty":
                sample["graph"]["edge_index"] = torch.empty((2, 0))
            case "complete":
                edges = torch.combinations(torch.arange(num_nodes), r=2).t()
                sample["graph"]["edge_index"] = torch.cat([edges, edges.flip(0)], dim=1)
            case "random":
                sample["graph"]["edge_index"] = ...
        base_token_probs = comp_token_probs(model, tok, sample)

        # Verbose output
        if args.verbose:
            print(f"\n======== Sample ID: {sample['index']} ========")
            print("- Prompt:", repr(sample["prompt"]))
            print("- Completion:", repr(sample["completion"]))
            print("Completion token probabilities:")
            for label, token_rows in [("w/ graph", org_token_probs), ("w/o graph", base_token_probs)]:
                print(f"\n-- {label} --")
                for token_id, token_str, log_prob, prob in token_rows:
                    print(f"{token_str}\t(id={token_id})\tprob={prob:.6g}\tlog_prob={log_prob:.6g}")

        # Plot probabilities
        plot_prob_comparison(
            org_token_probs,
            base_token_probs,
            output_path=os.path.join(OUT_DIR, f"tok_probs_{sample['index']}.png"),
        )


if __name__ == "__main__":
    main()
