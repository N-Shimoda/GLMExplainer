import argparse
import math
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from eval import create_pyg_batch  # noqa: E402
from src.ckpt import _resolve_ckpt_path  # noqa: E402
from src.constants import MOTIFQA_SUBSETS  # noqa: E402
from src.explanation.args import check_non_negative_int  # noqa: E402
from src.explanation.preprocess import build_dataset, filter_dataset  # noqa: E402
from src.glm import GraphTokenLM  # noqa: E402


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
        default="complete",
        choices=["complete", "empty", "random"],
        help="Type of baseline graph to use (default: complete).",
    )
    p.add_argument(
        "--output-dir", type=str, default="case_study", help="Directory to save output plots (default: case_study)."
    )
    p.add_argument("--verbose", action="store_true", help="If set, print token probabilities.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    return p.parse_args()


def _build_nx_graph(edge_index: torch.Tensor, num_nodes: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    if edge_index.numel() > 0:
        edges = edge_index.t().tolist()
        graph.add_edges_from((int(src), int(dst)) for src, dst in edges)
    return graph


def plot_prob_comparison(
    org_token_probs: list[tuple[int, str, float, float]],
    base_token_probs: list[tuple[int, str, float, float]],
    org_edge_index: torch.Tensor,
    base_edge_index: torch.Tensor,
    num_nodes: int,
    node_labels: Optional[list[int]] = None,
    output_path: str = "plots/token_prob_comparison.png",
):
    """Plot token probability comparison with original and baseline graphs.

    Parameters
    ----------
    org_token_probs : list[tuple[int, str, float, float]]
        Token rows for the original graph run, as (token_id, token_str, log_prob, prob).
    base_token_probs : list[tuple[int, str, float, float]]
        Token rows for the baseline graph run, as (token_id, token_str, log_prob, prob).
    org_edge_index : torch.Tensor
        Edge index for the original graph, shape (2, E).
    base_edge_index : torch.Tensor
        Edge index for the baseline graph, shape (2, E).
    num_nodes : int
        Number of nodes in both graphs.
    node_labels : list[int] | None, optional
        Optional node labels to render; length must match ``num_nodes`` when provided.
    output_path : str, optional
        Path to save the rendered figure.

    Returns
    -------
    None
        The figure is saved to ``output_path``.
    """
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

    # Build two graphs
    org_edge_index = torch.as_tensor(org_edge_index, dtype=torch.long)
    base_edge_index = torch.as_tensor(base_edge_index, dtype=torch.long)
    org_graph = _build_nx_graph(org_edge_index, num_nodes)
    base_graph = _build_nx_graph(base_edge_index, num_nodes)
    labels = None
    if node_labels is not None and len(node_labels) == num_nodes:
        labels = {idx: str(node_labels[idx]) for idx in range(num_nodes)}

    pos = nx.spring_layout(org_graph if org_graph.number_of_edges() else base_graph, seed=42) if num_nodes else {}

    # Plotting
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.2, 2.2])
    ax_org = fig.add_subplot(gs[0, 0])
    ax_base = fig.add_subplot(gs[0, 1])
    ax_prob = fig.add_subplot(gs[1, :])

    ax_org.set_title("Original graph")
    ax_base.set_title("Baseline graph")
    for ax, graph in [(ax_org, org_graph), (ax_base, base_graph)]:
        ax.axis("off")
        if num_nodes == 0:
            ax.text(0.5, 0.5, "Empty graph", ha="center", va="center")
            continue
        if graph.number_of_edges() == 0:
            ax.text(0.5, 0.5, "No edges", ha="center", va="center")
        nx.draw_networkx_edges(graph, pos, ax=ax, width=1.2, alpha=0.7)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=220, node_color="#87ceeb", edgecolors="#333333")
        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=8, font_color="white")

    xs = list(range(len(tokens)))
    ax_prob.set_ylim(0, 1.05)
    ax_prob.plot(xs, org_probs, marker="o", linewidth=1.5, label="w/ graph")
    ax_prob.plot(xs, base_probs, marker="x", linewidth=1.5, label="w/o graph")
    ax_prob.set_xticks(xs)
    ax_prob.set_xticklabels(tokens, rotation=40, ha="right")
    ax_prob.set_ylabel("Probability")
    ax_prob.set_title("Token Probability Comparison")
    ax_prob.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax_prob.legend()
    fig.tight_layout()
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
    """Run the case study comparing token probabilities with original vs. baseline graphs.

    This script loads a pre-trained GraphToken model and evaluates token prediction
    probabilities for completion text under two conditions:
    1. With the original graph structure from the dataset
    2. With a baseline graph (empty, complete, or random)

    The script processes samples from the MotifQA dataset, computes token probabilities
    for both graph conditions, and generates comparison plots showing how graph structure
    affects the model's token predictions.

    Side Effects
    ------------
    - Loads model checkpoint from disk
    - Creates output directory structure if it doesn't exist
    - Saves probability comparison plots to the output directory
    - Prints progress and optional verbose output to stdout

    Outputs
    -------
    PNG files
        Token probability comparison plots saved to:
        {output_dir}/{subset}/{baseline_graph}/tok_probs_{sample_index}.png

    Notes
    -----
    Command-line arguments control the behavior via parse_args():
    - --model-path: Path to the model checkpoint
    - --subset: MotifQA subset to use
    - --baseline-graph: Type of baseline graph (empty/complete/random)
    - --output-dir: Directory for saving plots
    - --verbose: Enable detailed probability output
    And other filtering/selection arguments.
    """
    args = parse_args()
    set_seed(args.seed)
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
        org_edge_index = sample["graph"]["edge_index"]
        org_token_probs = comp_token_probs(model, tok, sample)

        # Alternated input (work on a copy to avoid mutating the original sample)
        num_nodes = len(set(sample["nodes"]))
        base_sample = dict(sample)
        base_sample["graph"] = dict(sample["graph"])
        match args.baseline_graph:
            case "empty":
                base_sample["graph"]["edge_index"] = torch.empty((2, 0))
            case "complete":
                edges = torch.combinations(torch.arange(num_nodes), r=2).t()
                base_sample["graph"]["edge_index"] = torch.cat([edges, edges.flip(0)], dim=1)
            case "random":
                p = 0.3
                adj = torch.rand(num_nodes, num_nodes) < p
                adj = torch.triu(adj, diagonal=1)
                adj = adj + adj.t()
                base_sample["graph"]["edge_index"], _ = dense_to_sparse(adj)

        base_edge_index = base_sample["graph"]["edge_index"]
        base_token_probs = comp_token_probs(model, tok, base_sample)

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
            org_edge_index,
            base_edge_index,
            num_nodes,
            node_labels=sample["nodes"],
            output_path=os.path.join(OUT_DIR, f"tok_probs_{sample['index']}.png"),
        )


if __name__ == "__main__":
    main()
