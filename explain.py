import argparse
import os

import torch
from datasets import arrow_dataset, load_dataset
from torch_geometric.data import Batch as PygBatch
from torch_geometric.explain import Explainer, GNNExplainer
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig
from transformers.trainer_utils import set_seed

from eval import create_pyg_batch
from src.ckpt import _resolve_ckpt_path
from src.explanation import GLMWrapper
from src.glm import GraphTokenLM
from src.metrics import comp_accuracy
from src.preprocess import add_graph_column


def build_args():
    def check_non_negative_int(value: str) -> int:
        try:
            ivalue = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"`{value}` is not an integer.")
        if ivalue < 0:
            raise argparse.ArgumentTypeError("Value must be non-negative.")
        return ivalue

    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True, help="Path to the model checkpoint")
    p.add_argument(
        "--dataset",
        type=str,
        choices=["MotifQA", "GraphQA"],
        default="MotifQA",
        help="Dataset name (default: MotifQA)",
    )
    p.add_argument(
        "--subset",
        type=str,
        choices=["node_count", "edge_count", "cycle_check", "triangle_counting"],
        help="Dataset subset to use. Only applicable for GraphQA.",
    )
    p.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        default="test",
        help="Dataset split to use",
    )
    p.add_argument(
        "--target-value", type=check_non_negative_int, default=None, help="Targeted answer value to explain"
    )
    p.add_argument("--sample-idx", type=check_non_negative_int, default=None, help="Index of the sample to explain")
    p.add_argument("--num-trials", type=int, default=1, help="Number of trials for explaining each sample")

    args = p.parse_args()

    # Validate arguments
    if args.target_value is not None and args.sample_idx is not None:
        raise ValueError("Only one of `target_value` or `sample_idx` should be specified.")
    if args.dataset == "MotifQA" and args.subset is not None:
        raise ValueError("`subset` argument is only applicable for GraphQA dataset.")
    if args.dataset == "GraphQA" and args.subset is None:
        raise ValueError("`subset` argument must be specified for GraphQA dataset.")

    return args


def build_dataset(subset: str, dataset: str, split: str, node_feat_dim: int) -> arrow_dataset.Dataset:
    """Builds and returns the specified dataset subset and split."""
    match dataset:
        case "GraphQA":
            ds_raw = load_dataset("baharef/GraphQA", subset, split=f"zero_shot_{split}")
            ds = ds_raw.map(
                lambda x: add_graph_column(x, k=node_feat_dim, ds_name="GraphQA"),
                remove_columns=["algorithm", "answer", "nedges", "nnodes", "task_description", "text_encoding"],
                load_from_cache_file=False,
            )
        case "MotifQA":
            ds_raw = load_dataset("naos-ku/motif-qa", "yes_no", split=split)
            ds = ds_raw.map(
                lambda x: add_graph_column(x, k=node_feat_dim, ds_name="motif-qa"),
                remove_columns=["response", "nedges", "nnodes"],
                load_from_cache_file=False,
            )
    return ds.add_column("index", list(range(len(ds))))


def load_model(model_path: str) -> tuple[GraphTokenLM, AutoTokenizer]:
    """Loads the GraphTokenLM model and tokenizer from the specified checkpoint path.

    Parameters
    ----------
    model_path : str
        Path to the task directory, model directory or a specific checkpoint.

    Returns
    -------
    model : GraphTokenLM
        Loaded GraphTokenLM model.
    tokenizer : AutoTokenizer
        Corresponding tokenizer used with the model.
    """
    ckpt_path, run_name = _resolve_ckpt_path(model_path)

    model = GraphTokenLM.from_pretrained(ckpt_path, load_llm_weights=False)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loaded model from {ckpt_path} (run name: {run_name})")

    tokenizer = AutoTokenizer.from_pretrained(model.config.llm_name, trust_remote_code=True)
    return model, tokenizer


def get_gt_explanation(sample: dict[str, str]) -> torch.Tensor:
    """
    Build the binary ground-truth edge mask for a MotifQA sample.

    The dataset stores graphs using the canonical node ordering employed
    when constructing PyG objects. Edges that connect two nodes belonging
    to the 5-node house motif are marked with ``1``; all other edges are
    labeled ``0``. Because the explainer produces a bidirectional edge
    mask, both directions of each motif edge receive the positive label.

    Parameters
    ----------
    sample : dict[str, str]
        Dataset sample containing at least ``graph``, ``nodes``, ``edges``,
        and ``motif_nodes`` fields. ``graph`` must hold a PyG-compatible
        dictionary whose ``edge_index`` encodes the bidirectional edge list.

    Returns
    -------
    torch.Tensor
        One-dimensional tensor of length ``num_edges`` with binary entries
        indicating whether each edge in ``edge_index`` belongs to the motif.
    """
    graph_dict = sample["graph"]
    edge_index = torch.as_tensor(graph_dict["edge_index"], dtype=torch.long)
    num_edges = edge_index.size(1)
    gt_mask = torch.zeros(num_edges, dtype=torch.float)

    motif_nodes = sample["motif_nodes"]
    if len(motif_nodes) == 0 or num_edges == 0:
        return gt_mask

    nodes = sample["nodes"]
    edges = sample["edges"]
    node_to_idx = {nid: idx for idx, nid in enumerate(nodes)}

    # Build the undirected motif edge set using consecutive node indices.
    motif_edge_set: set[tuple[int, int]] = set()
    motif_node_set = set(motif_nodes)
    for u, v in edges:
        if u in motif_node_set and v in motif_node_set:
            if u not in node_to_idx or v not in node_to_idx:
                continue
            ui, vi = node_to_idx[u], node_to_idx[v]
            motif_edge_set.add(tuple(sorted((ui, vi))))

    if not motif_edge_set:
        return gt_mask

    for idx, (src, dst) in enumerate(edge_index.t().tolist()):
        if tuple(sorted((src, dst))) in motif_edge_set:
            gt_mask[idx] = 1.0

    return gt_mask


def explain_sample(
    wrapper: GLMWrapper, sample: dict[str, str], pyg_batch: PygBatch, gen_cfg: GenerationConfig, num_trials=10
):
    """Generates output for the given sample and explains it using GNNExplainer.

    Parameters
    ----------
    wrapper : GLMWrapper
        The model wrapper for GraphTokenLM.
    sample : dict
        A single dataset sample containing 'question' and 'completion'.
    pyg_batch : torch_geometric.data.Batch
        The graph data in PyG Batch format.
    gen_cfg : GenerationConfig
        Configuration for text generation.
    num_trials : int, optional
        Maximum number of trials to generate the correct answer, by default 10.

    Returns
    -------
    explanation : torch_geometric.explain.Explanation
        The explanation object containing the results.
    output_text : str
        The generated output text.
    """
    # Generate output and verify correctness
    generated = [wrapper.set_input(sample["prompt"], pyg_batch, gen_cfg) for _ in range(num_trials)]

    # Compute accuracy
    acc, _ = comp_accuracy(generated, [sample["completion"]] * len(generated), subset="house_check")
    print(f"Generated outputs: {generated} (acc={acc:.2f})")
    if acc < 1.0:
        print(
            f"[WARN] Failed to generate the correct answer after {num_trials} "
            f"trials (correct answer: {sample['completion']})."
        )
        return None, None

    # Generate explanation by GNNExplainer
    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=200),
        # algorithm=CaptumExplainer("IntegratedGradients"),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="graph",
            return_type="raw",
        ),
    )
    explanation = explainer(x=pyg_batch.x, edge_index=pyg_batch.edge_index, batch=pyg_batch.batch)
    return explanation, generated[0]


def main():
    set_seed(42)
    args = build_args()

    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path)
    model.eval()

    # Load dataset
    dataset = build_dataset(args.subset, args.dataset, args.split, node_feat_dim=model.config.node_feat_dim)
    # Filter dataset samples to explain
    match args.dataset:
        case "MotifQA":
            filtered_ds = dataset.select(range(2))
            OUT_DIR = os.path.join("explanations", "house_check")
        case "GraphQA":
            if args.target_value is not None:
                filtered_ds = dataset.filter(lambda x: int(x["completion"].split(".")[0]) == args.target_value)
                TARGET_VALUE = args.target_value
            elif args.sample_idx is not None:
                filtered_ds = dataset.filter(lambda x: x["index"] == args.sample_idx)
                TARGET_VALUE = int(filtered_ds[0]["completion"].split(".")[0])
            OUT_DIR = os.path.join("explanations", f"{args.subset}_{TARGET_VALUE}")

    print("Dataset: ", filtered_ds)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Create wrapper and generation config
    wrapper = GLMWrapper(model, tokenizer)
    gen_cfg = GenerationConfig(
        max_new_tokens=10,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Compute explanations for each sample
    for sample in tqdm(filtered_ds):
        for i in range(args.num_trials):
            pyg_batch = create_pyg_batch(sample["graph"], device=model.device)
            explanation, output_text = explain_sample(wrapper, sample, pyg_batch, gen_cfg)

            # Compute explanation accuracy
            gt_edge_mask = get_gt_explanation(sample)
            print(gt_edge_mask)

            if explanation is not None:
                print(f"Question: `{sample['prompt']}`")
                print(f"Generated answer: `{output_text}`")
                print(f"Correct answer: `{sample['completion']}`")
                print(f"Explanation: {explanation}")
                # Save explanation graphs
                suffix = f"{sample['index']}_{i}" if args.num_trials > 1 else f"{sample['index']}"
                graph_path = os.path.join(OUT_DIR, f"graph_{suffix}.svg")
                explanation.visualize_graph(graph_path)
                explanation.visualize_feature_importance(os.path.join(OUT_DIR, f"node_feat_{suffix}.svg"))
                print(f"Saved explanation graphs to {graph_path}")


if __name__ == "__main__":
    main()
