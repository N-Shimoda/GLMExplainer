import argparse
import csv
import os

import torch
from datasets import arrow_dataset, load_dataset
from torch_geometric.data import Batch as PygBatch
from torch_geometric.explain import Explainer, GNNExplainer, groundtruth_metrics
from torchmetrics.functional import average_precision
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
    p.add_argument("--num-samples", type=check_non_negative_int, default=None, help="Number of samples to explain")
    p.add_argument(
        "--target-value", type=check_non_negative_int, default=None, help="Targeted answer value to explain"
    )
    p.add_argument("--sample-idx", type=check_non_negative_int, default=None, help="Index of the sample to explain")
    p.add_argument("--num-trials", type=int, default=1, help="Number of trials for explaining each sample")
    p.add_argument(
        "--explain-pos-sample",
        action="store_true",
        help="If set, only explain positive samples (graphs containing house motifs).",
    )

    args = p.parse_args()

    # Validate arguments
    if args.target_value is not None and args.sample_idx is not None:
        raise ValueError("Only one of `target_value` or `sample_idx` should be specified.")
    if args.dataset == "MotifQA" and args.subset is not None:
        raise ValueError("`subset` argument is only applicable for GraphQA dataset.")
    if args.dataset == "GraphQA" and args.subset is None:
        raise ValueError("`subset` argument must be specified for GraphQA dataset.")
    if args.explain_pos_sample and args.dataset != "MotifQA":
        raise ValueError("`--explain-pos-sample` is only supported for the MotifQA dataset.")

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


def filter_dataset(
    dataset: arrow_dataset.Dataset,
    args: argparse.Namespace,
) -> tuple[arrow_dataset.Dataset, str]:
    match args.dataset:
        case "MotifQA":
            if args.explain_pos_sample:
                dataset = dataset.filter(lambda x: len(x["motif_nodes"]) > 0)
                print("Extracted positive samples: len(dataset) =", len(dataset))
            if args.num_samples is not None:
                num_to_select = min(args.num_samples, len(dataset))
                dataset = dataset.select(range(num_to_select))
            OUT_DIR = os.path.join("explanations", "house_check")
        case "GraphQA":
            if args.target_value is not None:
                dataset = dataset.filter(lambda x: int(x["completion"].split(".")[0]) == args.target_value)
                TARGET_VALUE = args.target_value
            elif args.sample_idx is not None:
                dataset = dataset.filter(lambda x: x["index"] == args.sample_idx)
                TARGET_VALUE = int(dataset[0]["completion"].split(".")[0])
            OUT_DIR = os.path.join("explanations", f"{args.subset}_{TARGET_VALUE}")

    return dataset, OUT_DIR


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


def _get_gt_explanation(sample: dict[str, str]) -> torch.Tensor:
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


def _generate_explanation(
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
    acc, _, correct_mask = comp_accuracy(generated, [sample["completion"]] * len(generated), subset="house_check")
    print(f"Generated outputs: {generated} (acc={acc:.2f})")
    if not any(correct_mask):
        print(
            f"[WARN] Failed to generate the correct answer after {num_trials} "
            f"trials (correct answer: {sample['completion']})."
        )
        return None, None, acc

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
    return explanation, generated[0], acc


def explain_sample(
    wrapper: GLMWrapper,
    sample: dict[str, str],
    trial_idx: int,
    num_trials: int,
    gen_cfg: GenerationConfig,
    log_path: str,
    fieldnames: list[str],
    dataset_name: str,
) -> tuple[bool, float, float, float]:
    """Explain a single sample, log metrics, and emit per-trial artifacts."""
    model_device = wrapper.model.device
    pyg_batch = create_pyg_batch(sample["graph"], device=model_device)
    explanation, output_text, acc = _generate_explanation(wrapper, sample, pyg_batch, gen_cfg)

    if explanation is None:
        return False, 0.0, 0.0, 0.0

    # Compute explanation accuracy
    gt_edge_mask = _get_gt_explanation(sample)
    pred_edge_mask = explanation.edge_mask.detach().cpu()
    auroc, f1 = groundtruth_metrics(pred_edge_mask, gt_edge_mask, metrics=["auroc", "f1_score"])
    auprc = average_precision(pred_edge_mask, gt_edge_mask.int(), task="binary").item()

    metrics_logged = False
    if dataset_name == "MotifQA" and len(sample.get("motif_nodes", [])) > 0:
        record = {
            "sample_index": sample["index"],
            "trial": trial_idx,
            "normal_accuracy": float(acc),
            "f1": float(f1),
            "auroc": float(auroc),
            "auprc": float(auprc),
        }
        with open(log_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(record)
        metrics_logged = True

    # Save explanation graphs
    suffix = f"{sample['index']}_{trial_idx}" if num_trials > 1 else f"{sample['index']}"
    out_dir = os.path.dirname(log_path)
    graph_path = os.path.join(out_dir, f"graph_{suffix}.svg")
    explanation.visualize_graph(graph_path)
    explanation.visualize_feature_importance(os.path.join(out_dir, f"node_feat_{suffix}.svg"))

    return metrics_logged, float(f1), float(auroc), float(auprc)


def main():
    set_seed(42)
    args = build_args()

    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path)
    model.eval()

    # Load dataset
    dataset = build_dataset(args.subset, args.dataset, args.split, node_feat_dim=model.config.node_feat_dim)
    dataset, OUT_DIR = filter_dataset(dataset, args)

    print("Dataset: ", dataset)
    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, "explanation_metrics.csv")
    fieldnames = ["sample_index", "trial", "normal_accuracy", "f1", "auroc", "auprc"]
    with open(log_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    total_f1 = 0.0
    total_auroc = 0.0
    total_auprc = 0.0
    total_count = 0

    # Create wrapper and generation config
    wrapper = GLMWrapper(model, tokenizer)
    gen_cfg = GenerationConfig(
        max_new_tokens=10,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Compute explanations for each sample
    for sample in tqdm(dataset):
        for i in range(args.num_trials):
            logged, f1, auroc, auprc = explain_sample(
                wrapper=wrapper,
                sample=sample,
                trial_idx=i,
                num_trials=args.num_trials,
                gen_cfg=gen_cfg,
                log_path=log_path,
                fieldnames=fieldnames,
                dataset_name=args.dataset,
            )

            if logged:
                total_f1 += f1
                total_auroc += auroc
                total_auprc += auprc
                total_count += 1

    if total_count > 0:
        avg_f1 = total_f1 / total_count
        avg_auroc = total_auroc / total_count
        avg_auprc = total_auprc / total_count
        print(
            "Average explanation accuracy across positive samples: "
            f"F1={avg_f1:.3f}, AUROC={avg_auroc:.3f}, AUPRC={avg_auprc:.3f}"
        )
        print(f"Saved explanation metrics to {log_path}")
    else:
        print("No explanation metrics recorded for positive samples.")


if __name__ == "__main__":
    main()
