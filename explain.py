import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime
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
from src.explanation.metrics import (
    EDGE_MASK_STABILITY_KEYS,
    compute_edge_mask_stability_metrics_per_sample,
)
from src.explanation.wrapper import GLMWrapper
from src.glm import GraphTokenLM
from src.metrics import comp_accuracy
from src.preprocess import add_graph_column
from src.utils import visualize_motif_explanation

GRAPH_SVG_SUBDIR = "graphs"
NODE_FEAT_SVG_SUBDIR = "node_feat"
TRIAL_OVERRIDE_COLUMN = "_trial_override"


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

    p = argparse.ArgumentParser(description="Explain GraphTokenLM predictions using GNNExplainer")
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
        help="Dataset split to use (default: test)",
    )
    p.add_argument(
        "--explain-pos-samples",
        action="store_true",
        help="If set, only explain positive samples (graphs containing house motifs).",
    )
    p.add_argument(
        "--num-samples", type=check_non_negative_int, default=None, help="Number of samples to explain (default: None)"
    )
    p.add_argument(
        "--target-value",
        type=check_non_negative_int,
        default=None,
        help="Targeted answer value to explain (default: None)",
    )
    p.add_argument(
        "--sample-idx",
        type=check_non_negative_int,
        default=None,
        help="Specify the index of the sample to explain (default: None)",
    )
    p.add_argument(
        "--num-trials", type=int, default=1, help="Number of trials for explaining each sample (default: 1)"
    )

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
    if args.num_samples is not None and args.sample_idx is not None:
        raise ValueError("Only one of `num_samples` or `sample_idx` should be specified.")
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
    # Define output directory
    subset = args.subset if args.subset is not None else "house_check"
    OUT_DIR = os.path.join("explanations", subset)

    # Filter dataset based on args
    if args.sample_idx is not None:
        dataset = dataset.filter(lambda x: x["index"] == args.sample_idx)

    match args.dataset:
        case "MotifQA":
            if args.explain_pos_samples:
                dataset = dataset.filter(lambda x: len(x["motif_nodes"]) > 0)
                print("Extracted positive samples: len(dataset) =", len(dataset))
        case "GraphQA":
            if args.target_value is not None:
                dataset = dataset.filter(lambda x: int(x["completion"].split(".")[0]) == args.target_value)
                TARGET_VALUE = args.target_value
            else:
                TARGET_VALUE = int(dataset[0]["completion"].split(".")[0])
            OUT_DIR = os.path.join("explanations", f"{args.subset}_{TARGET_VALUE}")

    if args.num_samples is not None:
        num_to_select = min(args.num_samples, len(dataset))
        dataset = dataset.select(range(num_to_select))

    return dataset, OUT_DIR


def _process_dataset(
    dataset: Iterable[dict[str, str]],
    args: argparse.Namespace,
    device: torch.device,
    log_path: str,
    fieldnames: list[str],
    show_progress: bool,
    model: GraphTokenLM,
    tokenizer: AutoTokenizer,
) -> tuple[
    float,
    float,
    float,
    float,
    int,
    dict[int, list[torch.Tensor]],
    dict[int, dict[str, float]],
]:
    """Run explanations across the dataset and accumulate trial-level metrics.

    Parameters
    ----------
    dataset : Iterable[dict[str, str]]
        Iterable of dataset samples. Each sample must at least expose the
        ``index`` key and any fields required by :func:`explain_sample`,
        including ``graph`` and model inputs.
    args : argparse.Namespace
        Parsed CLI arguments controlling trial repetition, dataset name, and
        optional per-sample overrides such as ``num_trials`` and ``dataset``.
    device : torch.device
        Target device on which the ``model`` should execute.
    log_path : str
        CSV path forwarded to :func:`explain_sample` for appending per-trial
        metrics.
    fieldnames : list[str]
        Ordered column names used by the CSV logger.
    show_progress : bool
        If ``True``, render a ``tqdm`` progress bar while processing samples.
    model : GraphTokenLM
        Pretrained GraphToken language model whose predictions are explained.
    tokenizer : AutoTokenizer
        Tokenizer paired with ``model`` and used to build prompts.

    Returns
    -------
    total_f1 : float
        Sum of F1 values over all trials that were successfully logged.
    total_auroc : float
        Sum of AUROC scores over the logged trials.
    total_auprc : float
        Sum of AUPRC scores over the logged trials.
    total_answer_accuracy : float
        Sum of answer accuracy scores over the logged trials.
    total_count : int
        Number of trials that produced metrics (i.e., were logged).
    sample_edge_masks : dict[int, list[torch.Tensor]]
        Mapping from sample index to the list of edge masks returned across its
        trials.
    sample_metrics : dict[int, dict[str, float]]
        Per-sample aggregates storing the sums of recorded metrics along with a
        ``count`` field.

    Notes
    -----
    If the dataset length is unknown, it is materialized into a list to enable
    progress reporting. When the dataset carries a ``_trial_override`` column,
    those overrides supersede ``args.num_trials`` for the affected samples.
    """
    if model.device != device:
        model = model.to(device)
    model.eval()

    gen_cfg = GenerationConfig(
        max_new_tokens=10,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    total_f1 = 0.0
    total_auroc = 0.0
    total_auprc = 0.0
    total_answer_accuracy = 0.0
    total_count = 0
    sample_edge_masks: defaultdict[int, list[torch.Tensor]] = defaultdict(list)
    sample_metrics: defaultdict[int, dict[str, float]] = defaultdict(
        lambda: {
            "answer_accuracy_sum": 0.0,
            "f1_sum": 0.0,
            "auroc_sum": 0.0,
            "auprc_sum": 0.0,
            "count": 0,
        }
    )
    try:
        dataset_length = len(dataset)  # type: ignore[arg-type]
    except TypeError:
        dataset = list(dataset)
        dataset_length = len(dataset)

    has_trial_override = False
    if hasattr(dataset, "column_names") and TRIAL_OVERRIDE_COLUMN in dataset.column_names:
        # Ensure at least one sample carries an override before switching modes.
        if dataset_length > 0 and dataset[0].get(TRIAL_OVERRIDE_COLUMN) is not None:
            has_trial_override = True

    per_sample_trials = 1 if has_trial_override else max(1, args.num_trials)
    total_steps = dataset_length * per_sample_trials
    progress = tqdm(total=total_steps) if show_progress and total_steps > 0 else None

    wrapper = GLMWrapper(model, tokenizer)

    for sample in dataset:
        sample_idx = sample["index"]
        override_value = sample.get(TRIAL_OVERRIDE_COLUMN) if has_trial_override else None
        if override_value is not None:
            trial_indices = [int(override_value)]
        else:
            trial_indices = range(args.num_trials)

        for i in trial_indices:
            (
                logged,
                f1,
                auroc,
                auprc,
                ans_accuracy_single,
                edge_mask,
            ) = explain_sample(
                wrapper=wrapper,
                sample=sample,
                trial_idx=i,
                num_trials=args.num_trials,
                gen_cfg=gen_cfg,
                log_path=log_path,
                fieldnames=fieldnames,
                dataset_name=args.dataset,
            )
            if edge_mask is not None:
                sample_edge_masks[sample_idx].append(edge_mask)
            if logged:
                total_f1 += f1
                total_auroc += auroc
                total_auprc += auprc
                total_answer_accuracy += ans_accuracy_single
                total_count += 1
                stats = sample_metrics[sample_idx]
                stats["answer_accuracy_sum"] += ans_accuracy_single
                stats["f1_sum"] += f1
                stats["auroc_sum"] += auroc
                stats["auprc_sum"] += auprc
                stats["count"] += 1
            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    return (
        total_f1,
        total_auroc,
        total_auprc,
        total_answer_accuracy,
        total_count,
        dict(sample_edge_masks),
        {idx: dict(stats) for idx, stats in sample_metrics.items()},
    )


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
            f"[WARN] Failed to generate the correct answer for sample[index={sample['index']}] "
            f"(correct answer: `{sample['completion']}`)."
        )
        return None, acc

    # Generate explanation by GNNExplainer
    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(
            epochs=200,
            lr=0.01,
            edge_size=24,
            edge_ent=2.0,
            num_hops=wrapper.model.config.num_gnn_layers,
        ),
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
) -> tuple[bool, float, float, float, float, torch.Tensor | None]:
    """Explain a single sample, log metrics, and emit per-trial artifacts."""
    model_device = wrapper.model.device
    pyg_batch = create_pyg_batch(sample["graph"], device=model_device)
    explanation, ans_accuracy = _generate_explanation(wrapper, sample, pyg_batch, gen_cfg, num_trials=NUM_GEN_TRIALS)

    if explanation is None:
        return False, 0.0, 0.0, 0.0, 0.0, None

    # Compute explanation accuracy
    gt_edge_mask = _get_gt_explanation(sample)
    pred_edge_mask = explanation.edge_mask.detach().cpu().float()
    auroc, f1 = groundtruth_metrics(pred_edge_mask, gt_edge_mask, metrics=["auroc", "f1_score"])
    auprc = average_precision(pred_edge_mask, gt_edge_mask.int(), task="binary").item()

    # Log explanation accuracy for positive samples
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

    # Directories to save figures
    suffix = f"{sample['index']}_{trial_idx}" if num_trials > 1 else f"{sample['index']}"
    out_dir = os.path.dirname(log_path)
    graph_dir = os.path.join(out_dir, GRAPH_SVG_SUBDIR, f"graph_{sample['index']}")
    node_feat_dir = os.path.join(out_dir, NODE_FEAT_SVG_SUBDIR, f"node_feat_{sample['index']}")
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(node_feat_dir, exist_ok=True)

    # Save visualizations
    graph_path = os.path.join(graph_dir, f"{suffix}.svg")
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
    feature_path = os.path.join(node_feat_dir, f"node_feat_{suffix}.svg")
    explanation.visualize_feature_importance(feature_path)

    return metrics_logged, float(f1), float(auroc), float(auprc), float(ans_accuracy), pred_edge_mask


def main():
    set_seed(42)
    args = build_args()
    run_name = datetime.now().strftime("%m%d-%H%M")

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
    if len(dataset) == 0:
        if is_rank0:
            print("[INFO] No samples to explain after filtering. Exiting.")
        if is_distributed:
            _cleanup_distributed()
        return

    # Duplicate a sample for multiple trials if len(dataset) == 1
    if args.num_trials > 1 and len(dataset) == 1:
        duplicate_indices = [0] * args.num_trials
        dataset = dataset.select(duplicate_indices)
        dataset = dataset.add_column(TRIAL_OVERRIDE_COLUMN, list(range(args.num_trials)))
    elif hasattr(dataset, "column_names") and TRIAL_OVERRIDE_COLUMN in dataset.column_names:
        dataset = dataset.remove_columns([TRIAL_OVERRIDE_COLUMN])

    if is_rank0:
        print("[INFO] Dataset: ", dataset)

    if is_distributed:
        dataset_len = len(dataset)
        if dataset_len >= world_size:
            dataset = dataset.shard(num_shards=world_size, index=rank)
        else:
            # Assign at most one sample per rank when the filtered dataset is
            # smaller than the world size to avoid out-of-range shard errors.
            indices = list(range(rank, dataset_len, world_size))
            dataset = dataset.select(indices if indices else [])

    os.makedirs(OUT_DIR, exist_ok=True)
    base_log_path = os.path.join(OUT_DIR, f"metrics_{run_name}.csv")
    shard_log_path = base_log_path if world_size == 1 else os.path.join(OUT_DIR, f"metrics_rank{rank}.csv")

    fieldnames = ["sample_index", "trial", "answer_accuracy", "f1", "auroc", "auprc"]
    _write_metrics_header(shard_log_path, fieldnames)

    show_progress = is_rank0 and len(dataset) > 0
    (
        total_f1,
        total_auroc,
        total_auprc,
        total_answer_accuracy,
        total_count,
        sample_edge_masks,
        sample_metrics,
    ) = _process_dataset(
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
        [total_f1, total_auroc, total_auprc, total_answer_accuracy, float(total_count)],
        device=device,
    )
    if is_distributed:
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    total_f1, total_auroc, total_auprc, total_answer_accuracy, total_count = metrics_tensor.tolist()
    total_count = int(total_count)

    if is_distributed:
        dist.barrier()
        if is_rank0:
            with open(base_log_path, "w", newline="") as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                for idx in range(world_size):
                    part_path = os.path.join(OUT_DIR, f"metrics_rank{idx}.csv")
                    if not os.path.exists(part_path):
                        continue
                    with open(part_path, newline="") as part_file:
                        reader = csv.DictReader(part_file)
                        for row in reader:
                            writer.writerow(row)
                    os.remove(part_path)
        dist.barrier()

    merged_edge_masks: dict[int, list[torch.Tensor]] | None
    merged_sample_metrics: dict[int, dict[str, float]] | None
    if is_distributed:
        gathered_masks = [None] * world_size if is_rank0 else None
        dist.gather_object(sample_edge_masks, gathered_masks, dst=0)
        gathered_metrics = [None] * world_size if is_rank0 else None
        dist.gather_object(sample_metrics, gathered_metrics, dst=0)
        if is_rank0:
            merged = defaultdict(list)
            merged_metrics_accum = defaultdict(
                lambda: {
                    "answer_accuracy_sum": 0.0,
                    "f1_sum": 0.0,
                    "auroc_sum": 0.0,
                    "auprc_sum": 0.0,
                    "count": 0,
                }
            )
            assert gathered_masks is not None
            for partial in gathered_masks:
                if not partial:
                    continue
                for idx, masks in partial.items():
                    merged[idx].extend(masks)
            merged_edge_masks = dict(merged)
            assert gathered_metrics is not None
            for partial_metrics in gathered_metrics:
                if not partial_metrics:
                    continue
                for idx, stats in partial_metrics.items():
                    acc = merged_metrics_accum[idx]
                    acc["answer_accuracy_sum"] += stats.get("answer_accuracy_sum", 0.0)
                    acc["f1_sum"] += stats.get("f1_sum", 0.0)
                    acc["auroc_sum"] += stats.get("auroc_sum", 0.0)
                    acc["auprc_sum"] += stats.get("auprc_sum", 0.0)
                    acc["count"] += stats.get("count", 0)
            merged_sample_metrics = {idx: dict(vals) for idx, vals in merged_metrics_accum.items()}
        else:
            merged_edge_masks = None
            merged_sample_metrics = None
    else:
        merged_edge_masks = sample_edge_masks
        merged_sample_metrics = sample_metrics

    if is_rank0:
        if total_count > 0:
            avg_answer_accuracy = total_answer_accuracy / total_count
            avg_f1 = total_f1 / total_count
            avg_auroc = total_auroc / total_count
            avg_auprc = total_auprc / total_count
            print(
                "Average explanation accuracy across positive samples: "
                f"AnswerAcc={avg_answer_accuracy:.3f}, "
                f"F1={avg_f1:.3f}, AUROC={avg_auroc:.3f}, AUPRC={avg_auprc:.3f}"
            )
            print(f"Saved explanation metrics to {base_log_path}")
        else:
            avg_answer_accuracy = 0.0
            print("No explanation metrics recorded for positive samples.")

        avg_metrics_path = os.path.join(OUT_DIR, f"average_metrics_{run_name}.csv")
        average_fieldnames = [
            "sample_index",
            "answer_accuracy",
            "f1",
            "auroc",
            "auprc",
            *EDGE_MASK_STABILITY_KEYS,
        ]
        per_sample_stability = (
            compute_edge_mask_stability_metrics_per_sample(merged_edge_masks or {})
            if merged_edge_masks is not None
            else {}
        )
        sample_metrics_for_logging = merged_sample_metrics or {}
        zero_stability = {key: 0.0 for key in EDGE_MASK_STABILITY_KEYS}
        with open(avg_metrics_path, "w", newline="") as avg_file:
            writer = csv.DictWriter(avg_file, fieldnames=average_fieldnames)
            writer.writeheader()
            for sample_idx in sorted(sample_metrics_for_logging.keys()):
                stats = sample_metrics_for_logging[sample_idx]
                count = int(stats.get("count", 0))
                answer_acc_sum = stats.get("answer_accuracy_sum", 0.0)
                f1_sum = stats.get("f1_sum", 0.0)
                auroc_sum = stats.get("auroc_sum", 0.0)
                auprc_sum = stats.get("auprc_sum", 0.0)
                row = {
                    "sample_index": sample_idx,
                    "answer_accuracy": answer_acc_sum / count if count > 0 else 0.0,
                    "f1": f1_sum / count if count > 0 else 0.0,
                    "auroc": auroc_sum / count if count > 0 else 0.0,
                    "auprc": auprc_sum / count if count > 0 else 0.0,
                }
                stability = per_sample_stability.get(sample_idx, zero_stability)
                for key in EDGE_MASK_STABILITY_KEYS:
                    row[key] = stability.get(key, 0.0)
                writer.writerow(row)
        print(f"Saved average metrics to {avg_metrics_path}")

    if is_distributed:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
