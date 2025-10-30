import argparse
import csv
import os
from typing import Iterable

import torch
import torch.distributed as dist
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
from src.utils import visualize_motif_explanation


def _write_metrics_header(log_path: str, fieldnames: list[str]) -> None:
    with open(log_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def _init_distributed_if_needed() -> tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1, 0, False

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world_size, local_rank, True


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


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
        "--explain-pos-samples",
        action="store_true",
        help="If set, only explain positive samples (graphs containing house motifs).",
    )
    p.add_argument("--num-samples", type=check_non_negative_int, default=None, help="Number of samples to explain")
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
    if args.explain_pos_samples and args.dataset != "MotifQA":
        raise ValueError("`--explain-pos-sample` is only supported for the MotifQA dataset.")
    if args.num_trials < 1:
        raise ValueError("`num_trials` must be at least 1.")

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
            if args.explain_pos_samples:
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


def load_model(
    model_path: str, device: torch.device | str | None = None, verbose: bool = True
) -> tuple[GraphTokenLM, AutoTokenizer]:
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

    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available.")
        if target_device.type == "cuda" and target_device.index is not None:
            torch.cuda.set_device(target_device.index)

    model = GraphTokenLM.from_pretrained(ckpt_path, load_llm_weights=False)
    model.to(target_device)
    if verbose:
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
) -> tuple[torch.Tensor | None, float]:
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
    explanation : torch_geometric.explain.Explanation | None
        The explanation object containing the results, or ``None`` when no
        correct answer was produced within the allotted trials.
    accuracy : float
        Ratio of correct generations within ``num_trials``.
    """
    # Generate output and verify correctness
    generated = [wrapper.set_input(sample["prompt"], pyg_batch, gen_cfg) for _ in range(num_trials)]

    # Compute accuracy
    acc, _, correct_mask = comp_accuracy(generated, [sample["completion"]] * len(generated), subset="house_check")
    if not any(correct_mask):
        print(
            f"[WARN] Failed to generate the correct answer after {num_trials} trials "
            f"(correct answer: {sample['completion']})."
        )
        return None, acc

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
    return explanation, acc


def explain_sample(
    wrapper: GLMWrapper,
    sample: dict[str, str],
    trial_idx: int,
    num_trials: int,
    gen_cfg: GenerationConfig,
    log_path: str,
    fieldnames: list[str],
    dataset_name: str,
    NUM_GEN_TRIALS: int = 10,
) -> tuple[bool, float, float, float]:
    """Explain a single sample, log metrics, and emit per-trial artifacts."""
    model_device = wrapper.model.device
    pyg_batch = create_pyg_batch(sample["graph"], device=model_device)
    explanation, ans_accuracy = _generate_explanation(wrapper, sample, pyg_batch, gen_cfg, num_trials=NUM_GEN_TRIALS)

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
            "answer_accuracy": float(ans_accuracy),
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
    if dataset_name == "MotifQA":
        visualize_motif_explanation(
            sample=sample,
            explanation=explanation,
            graph_path=graph_path,
            f1=float(f1),
            auroc=float(auroc),
            auprc=float(auprc),
            ans_accuracy=float(ans_accuracy),
        )
    else:
        explanation.visualize_graph(graph_path)
    feature_path = os.path.join(out_dir, f"node_feat_{suffix}.svg")
    explanation.visualize_feature_importance(feature_path)

    return metrics_logged, float(f1), float(auroc), float(auprc)


def _process_dataset(
    dataset: Iterable[dict[str, str]],
    args: argparse.Namespace,
    device: torch.device,
    log_path: str,
    fieldnames: list[str],
    show_progress: bool,
    model: GraphTokenLM,
    tokenizer: AutoTokenizer,
) -> tuple[float, float, float, int]:
    if model.device != device:
        model = model.to(device)
    model.eval()

    wrapper = GLMWrapper(model, tokenizer)
    gen_cfg = GenerationConfig(
        max_new_tokens=10,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    total_f1 = 0.0
    total_auroc = 0.0
    total_auprc = 0.0
    total_count = 0
    try:
        dataset_length = len(dataset)  # type: ignore[arg-type]
    except TypeError:
        dataset = list(dataset)
        dataset_length = len(dataset)
    total_steps = dataset_length * max(1, args.num_trials)
    progress = tqdm(total=total_steps) if show_progress and total_steps > 0 else None

    for sample in dataset:
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
            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    return total_f1, total_auroc, total_auprc, total_count


def main():
    set_seed(42)
    args = build_args()

    rank, world_size, local_rank, is_distributed = _init_distributed_if_needed()
    set_seed(42 + rank)

    if is_distributed and not torch.cuda.is_available():
        raise RuntimeError("Distributed execution requires CUDA devices.")

    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_rank0 = rank == 0

    model, tokenizer = load_model(args.model_path, device=device, verbose=is_rank0)
    model.eval()

    node_feat_dim = model.config.node_feat_dim
    dataset = build_dataset(args.subset, args.dataset, args.split, node_feat_dim=node_feat_dim)
    dataset, OUT_DIR = filter_dataset(dataset, args)
    if is_rank0:
        print("Dataset: ", dataset)

    if is_distributed:
        dataset = dataset.shard(num_shards=world_size, index=rank)

    os.makedirs(OUT_DIR, exist_ok=True)
    base_log_path = os.path.join(OUT_DIR, "explanation_metrics.csv")
    shard_log_path = base_log_path if world_size == 1 else os.path.join(OUT_DIR, f"explanation_metrics_rank{rank}.csv")

    fieldnames = ["sample_index", "trial", "answer_accuracy", "f1", "auroc", "auprc"]
    _write_metrics_header(shard_log_path, fieldnames)

    show_progress = is_rank0 and len(dataset) > 0
    total_f1, total_auroc, total_auprc, total_count = _process_dataset(
        dataset=dataset,
        args=args,
        device=device,
        log_path=shard_log_path,
        fieldnames=fieldnames,
        show_progress=show_progress,
        model=model,
        tokenizer=tokenizer,
    )

    metrics_tensor = torch.tensor(
        [total_f1, total_auroc, total_auprc, float(total_count)],
        device=device,
    )
    if is_distributed:
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    total_f1, total_auroc, total_auprc, total_count = metrics_tensor.tolist()
    total_count = int(total_count)

    if is_distributed:
        dist.barrier()
        if is_rank0:
            with open(base_log_path, "w", newline="") as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                for idx in range(world_size):
                    part_path = os.path.join(OUT_DIR, f"explanation_metrics_rank{idx}.csv")
                    if not os.path.exists(part_path):
                        continue
                    with open(part_path, newline="") as part_file:
                        reader = csv.DictReader(part_file)
                        for row in reader:
                            writer.writerow(row)
                    os.remove(part_path)
        dist.barrier()

    if is_rank0:
        if total_count > 0:
            avg_f1 = total_f1 / total_count
            avg_auroc = total_auroc / total_count
            avg_auprc = total_auprc / total_count
            print(
                "Average explanation accuracy across positive samples: "
                f"F1={avg_f1:.3f}, AUROC={avg_auroc:.3f}, AUPRC={avg_auprc:.3f}"
            )
            print(f"Saved explanation metrics to {base_log_path}")
        else:
            print("No explanation metrics recorded for positive samples.")

    if is_distributed:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
