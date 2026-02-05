import argparse
import math
import os
import sys
from typing import Literal, Optional

import matplotlib as mpl
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
        "--output-dir",
        type=str,
        default="plots/llr_study",
        help="Directory to save output plots (default: plots/llr_study).",
    )
    p.add_argument(
        "--output-format",
        type=str,
        default="pdf",
        choices=["pdf", "svg", "png"],
        help="Output plot format (default: pdf).",
    )
    p.add_argument("--verbose", action="store_true", help="If set, print token probabilities.")
    p.add_argument(
        "--llr-threshold",
        type=float,
        default=1.0,
        help="Absolute LLR threshold to highlight token labels (default: 1.0).",
    )
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
    baseline_graph: Literal["complete", "empty", "random"],
    num_nodes: int,
    node_labels: Optional[list[int]] = None,
    llr_rows: Optional[list[tuple[str, int, float]]] = None,
    llr_threshold: float = 1.0,
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
    baseline_graph: Literal["complete", "empty", "random"]
        Type of baseline graph used.
    num_nodes : int
        Number of nodes in both graphs.
    node_labels : list[int] | None, optional
        Optional node labels to render; length must match ``num_nodes`` when provided.
    llr_rows : list[tuple[str, int, float]] | None, optional
        Optional LLR rows as (token_str, token_id, llr). If not provided, LLRs
        are computed from the provided probabilities.
    llr_threshold : float, optional
        Absolute LLR threshold above which token labels are highlighted, by default 1.0.
    output_path : str, optional
        Path to save the rendered figure.
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
    if llr_rows is None:
        llr_rows, _ = comp_log_likelihood_ratio(org_token_probs, base_token_probs)
    if len(llr_rows) != len(tokens):
        raise ValueError("LLR rows length must match token probabilities length.")
    for (llr_token, llr_id, _), org_row in zip(llr_rows, org_token_probs):
        org_id, org_token, _, _ = org_row
        if llr_id != org_id or llr_token != org_token:
            raise ValueError("LLR token sequence does not match token probabilities.")
    llrs = [llr for _, _, llr in llr_rows]

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
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.2, 2.2], hspace=0.35)
    ax_org = fig.add_subplot(gs[0, 0])
    ax_base = fig.add_subplot(gs[0, 1])
    ax_prob = fig.add_subplot(gs[1, :])

    ax_org.set_title("Original graph", fontsize=15)
    ax_base.set_title(f"{baseline_graph.capitalize()} graph", fontsize=15)
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
    ax_prob.plot(xs, org_probs, marker="o", linewidth=1.5, label="Original graph")
    ax_prob.plot(xs, base_probs, marker="x", linewidth=1.5, label=f"{baseline_graph.capitalize()} graph")
    ax_prob.set_xticks(xs)
    ax_prob.set_xticklabels(tokens, rotation=40, ha="right", fontsize=15)
    ax_prob.tick_params(axis="y", labelsize=14)
    ax_prob.set_ylabel("Probability", fontsize=14)
    ax_prob.set_title("Token Probability Comparison", fontsize=15)
    ax_prob.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax_prob.legend(fontsize=14)

    # Highlight tokens with high absolute LLR
    for label, llr in zip(ax_prob.get_xticklabels(), llrs):
        if abs(llr) > llr_threshold:
            label.set_color("#d62728")

    # Keep text as text in SVG
    suffix = output_path.split(".")[-1].lower()
    match suffix:
        case "svg":
            mpl.rcParams["svg.fonttype"] = "none"
        case "pdf":
            mpl.rcParams["pdf.fonttype"] = 42

    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_llr_histograms(
    llr_by_token: dict[tuple[str, int], list[float]],
    output_path: str,
    bins: int = 30,
    llr_threshold: float = 1.0,
):
    """Plot per-token LLR histograms aggregated across samples.

    Parameters
    ----------
    llr_by_token : dict[tuple[str, int], list[float]]
        Mapping from (token_str, token_id) to list of LLR values.
    output_path : str
        Path to save the rendered figure.
    bins : int, optional
        Number of histogram bins, by default 30.
    """
    if not llr_by_token:
        raise ValueError("No LLR values provided for histogram plot.")

    tokens = list(llr_by_token.keys())
    num_tokens = len(tokens)
    ncols = min(3, num_tokens)
    nrows = math.ceil(num_tokens / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 2.8 * nrows))
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten().tolist()

    for ax, token_key in zip(axes, tokens):
        token_str, token_id = token_key
        llrs = llr_by_token[token_key]
        ratio_over_threshold = 0.0
        if llrs:
            ratio_over_threshold = sum(1 for llr in llrs if llr > llr_threshold) / len(llrs)
        ax.hist(llrs, bins=bins, color="#4c72b0", alpha=0.85)
        ax.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.0)
        ax.set_title(
            f"{token_str} ({token_id})  >{llr_threshold:g}: {ratio_over_threshold:.2f}",
            fontsize=11,
        )
        ax.set_xlabel("LLR", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    for ax in axes[len(tokens) :]:
        ax.axis("off")

    plt.tight_layout()

    suffix = output_path.split(".")[-1].lower()
    match suffix:
        case "svg":
            mpl.rcParams["svg.fonttype"] = "none"
        case "pdf":
            mpl.rcParams["pdf.fonttype"] = 42

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


def comp_log_likelihood_ratio(
    org_token_probs: list[tuple[int, str, float, float]],
    base_token_probs: list[tuple[int, str, float, float]],
) -> tuple[list[tuple[str, int, float]], float]:
    """Compute per-token log likelihood ratios between two runs.

    Parameters
    ----------
    org_token_probs : list[tuple[int, str, float, float]]
        Original run token probabilities as (token_id, token_str, log_prob, prob).
    base_token_probs : list[tuple[int, str, float, float]]
        Baseline run token probabilities as (token_id, token_str, log_prob, prob).

    Returns
    -------
    list[tuple[str, int, float]]
        Per-token rows as (token_str, token_id, llr).
    float
        Total log likelihood ratio across tokens.

    Notes
    -----
    Log likelihood ratio (LLR) for each token is computed as:
        LLR(token) = log_prob_org(token) - log_prob_base(token)
    """
    if len(org_token_probs) != len(base_token_probs):
        raise ValueError("Token probability lists must be the same length for LLR.")

    llr_rows = []
    total_llr = 0.0
    for org_row, base_row in zip(org_token_probs, base_token_probs):
        org_id, org_token, org_log_prob, _ = org_row
        base_id, base_token, base_log_prob, _ = base_row
        if org_id != base_id or org_token != base_token:
            raise ValueError("Token sequences do not match between runs for LLR.")
        llr = org_log_prob - base_log_prob
        llr_rows.append((org_token, org_id, llr))
        total_llr += llr

    return llr_rows, total_llr


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
    lpe_dim = getattr(model.config, "lpe_dim", model.config.node_feat_dim)
    use_degree_emb = getattr(model.config, "use_degree_emb", False)
    dataset = build_dataset("MotifQA", args.subset, "test", lpe_dim=lpe_dim, use_degree_emb=use_degree_emb)
    dataset = filter_dataset(
        dataset,
        "MotifQA",
        target_pos_samples=args.target_pos_samples,
        sample_idx=args.sample_idx,
        num_samples=args.num_samples,
    )

    llr_by_token: dict[tuple[str, int], list[float]] = {}
    llr_ratio_by_sample: list[float] = []

    for sample in tqdm(dataset, desc="Processing samples", disable=args.verbose):
        # Original input
        org_edge_index = sample["graph"]["edge_index"]
        org_token_probs = comp_token_probs(model, tok, sample)

        # Alternated input
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
        llr_rows, total_llr = comp_log_likelihood_ratio(org_token_probs, base_token_probs)
        for token_str, token_id, llr in llr_rows:
            llr_by_token.setdefault((token_str, token_id), []).append(llr)
        if llr_rows:
            ratio_over_threshold = sum(1 for _, _, llr in llr_rows if llr > args.llr_threshold) / len(llr_rows)
            llr_ratio_by_sample.append(ratio_over_threshold)

        # Verbose output
        if args.verbose:
            print(f"\n======== Sample ID: {sample['index']} ========")
            print("- Prompt:", repr(sample["prompt"]))
            print("- Completion:", repr(sample["completion"]))
            print("\nCompletion token probabilities (incl. EOS):")
            print(f"{'Token':<8} {'ID':>6} {'Org prob':>10} {'Base prob':>10} {'LLR':>10}")
            print("-" * 50)
            for (token_str, token_id, llr), org_row, base_row in zip(llr_rows, org_token_probs, base_token_probs):
                org_prob = org_row[3]
                base_prob = base_row[3]
                print(f"{token_str:<8} {token_id:>6} {org_prob:>10.4g} {base_prob:>10.4g} {llr:>10.4g}")
            print(f"\nTotal LLR: {total_llr:.4g}")
            if llr_rows:
                print(
                    f"Ratio LLR > {args.llr_threshold:g}: "
                    f"{ratio_over_threshold:.3f} ({ratio_over_threshold * 100:.1f}%)"
                )

        # Plot probabilities
        plot_prob_comparison(
            org_token_probs,
            base_token_probs,
            org_edge_index,
            base_edge_index,
            args.baseline_graph,
            num_nodes,
            node_labels=sample["nodes"],
            llr_rows=llr_rows,
            llr_threshold=args.llr_threshold,
            output_path=os.path.join(OUT_DIR, f"tok_probs_{sample['index']}.{args.output_format}"),
        )

    if llr_by_token:
        plot_llr_histograms(
            llr_by_token,
            output_path=os.path.join(OUT_DIR, f"llr_histograms.{args.output_format}"),
            llr_threshold=args.llr_threshold,
        )
    if llr_ratio_by_sample:
        avg_ratio = sum(llr_ratio_by_sample) / len(llr_ratio_by_sample)
        print(
            f"\nAverage ratio LLR > {args.llr_threshold:g} across samples: "
            f"{avg_ratio:.3f} ({avg_ratio * 100:.1f}%)"
        )


if __name__ == "__main__":
    main()
