import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime
from typing import Iterable

import torch
import torch.distributed as dist
from torch_geometric.data import Batch as PygBatch
from torch_geometric.explain import (
    Explainer,
    Explanation,
    GNNExplainer,
    groundtruth_metrics,
)
from torchmetrics.functional import average_precision
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig
from transformers.trainer_utils import set_seed

import wandb
from eval import create_pyg_batch
from src.ckpt import _resolve_ckpt_path
from src.explanation.logging import _write_metrics_header, write_average_metrics_csv
from src.explanation.preprocess import build_dataset, filter_dataset
from src.explanation.wrapper import GLMWrapper
from src.glm import GraphTokenLM
from src.metrics import comp_accuracy
from src.utils import visualize_motif_explanation

GRAPH_SVG_SUBDIR = "graphs"
NODE_FEAT_SVG_SUBDIR = "node_feat"
TRIAL_OVERRIDE_COLUMN = "_trial_override"


def _init_distributed_if_needed() -> tuple[int, int, int, bool]:
    """Initialize torch.distributed and return rank metadata if WORLD_SIZE > 1.

    Returns:
        A tuple of (rank, world_size, local_rank, initialized).
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1, 0, False

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    local_rank_env = os.environ.get("LOCAL_RANK")
    device_id: int | None = None
    if torch.cuda.is_available():
        try:
            device_id = int(local_rank_env) if local_rank_env is not None else torch.cuda.current_device()
        except ValueError:
            device_id = torch.cuda.current_device()

    init_kwargs: dict[str, object] = {"backend": backend}
    if backend == "nccl" and device_id is not None:
        init_kwargs["device_id"] = device_id

    dist.init_process_group(**init_kwargs)
    rank = dist.get_rank()
    try:
        local_rank = int(local_rank_env) if local_rank_env is not None else rank
    except ValueError:
        local_rank = rank
    return rank, world_size, local_rank, True


def _cleanup_distributed() -> None:
    """Destroy the torch.distributed process group if it is currently active."""
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

    # Dataset arguments
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

    # Dataset filtering arguments
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

    # GNNExplainer arguments
    p.add_argument("--edge-size", type=float, default=0.005, help="GNNExplainer edge size parameter (default: 0.005)")
    p.add_argument("--edge-ent", type=float, default=1.0, help="GNNExplainer edge entropy parameter (default: 1.0)")
    p.add_argument("--epochs", type=int, default=200, help="GNNExplainer optimization epochs (default: 200)")
    p.add_argument("--lr", type=float, default=0.01, help="GNNExplainer learning rate (default: 0.01)")

    # Logging arguments
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Log per-sample explanation metrics to Weights & Biases.",
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


def load_model(
    model_path: str, device: torch.device | str, verbose: bool = True
) -> tuple[GraphTokenLM, AutoTokenizer]:
    """Loads the GraphTokenLM model and tokenizer from the specified checkpoint path.

    Parameters
    ----------
    model_path : str
        Path to the task directory, model directory or a specific checkpoint.
    device : torch.device | str
        Target device to which the model is moved after loading.
    verbose : bool, optional
        If ``True``, prints loading information, by default True.

    Returns
    -------
    model : GraphTokenLM
        Loaded GraphTokenLM model.
    tokenizer : AutoTokenizer
        Corresponding tokenizer used with the model.
    """
    # Resolve checkpoint path
    ckpt_path, run_name = _resolve_ckpt_path(model_path)

    # Load model onto the specified device
    model = GraphTokenLM.from_pretrained(ckpt_path, load_llm_weights=False)
    model.to(torch.device(device))
    if verbose:
        print(f"Loaded model from {ckpt_path} (run name: {run_name})")

    # Load pre-trained tokenizer
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

    node_to_idx = {nid: idx for idx, nid in enumerate(sample["nodes"])}

    # Build the undirected motif edge set using consecutive node indices.
    motif_edge_set: set[tuple[int, int]] = set()
    motif_node_set = set(motif_nodes)
    for u, v in sample["edges"]:
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
    wrapper: GLMWrapper,
    sample: dict[str, str],
    pyg_batch: PygBatch,
    gen_cfg: GenerationConfig,
    explainer_args: dict[str, float | int],
    num_trials: int = 10,
) -> tuple[Explanation | None, float]:
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
    explainer_args : dict[str, float | int]
        Keyword arguments forwarded to :class:`GNNExplainer` controlling its optimization.
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
        algorithm=GNNExplainer(num_hops=wrapper.model.config.num_gnn_layers, **explainer_args),
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
    explainer_args: dict[str, float | int],
    num_gen_trials: int = 10,
) -> tuple[bool, dict[str, float], float, torch.Tensor | None]:
    """Explain a single dataset sample and collect per-trial artifacts.

    Parameters
    ----------
    wrapper : GLMWrapper
        Wrapper around the GraphToken language model that exposes convenience
        methods for generation and explanation.
    sample : dict[str, str]
        Dataset entry that must contain the graph structure as well as fields
        required by :func:`create_pyg_batch` and :func:`_generate_explanation`.
    trial_idx : int
        Index of the current trial for the given ``sample``.
    num_trials : int
        Total number of trials that will be executed for the ``sample``.
    gen_cfg : GenerationConfig
        Configuration controlling the language-model generation step.
    log_path : str
        CSV destination to which per-trial metrics are appended.
    fieldnames : list[str]
        Column ordering used when writing to ``log_path``.
    dataset_name : str
        Name of the dataset being processed (e.g., ``"MotifQA"``).
    explainer_args : dict[str, float | int]
        Keyword arguments forwarded to the explainer factory.
    num_gen_trials : int, default=10
        Maximum number of explanation generation trials passed to
        :func:`_generate_explanation`.

    Returns
    -------
    metrics_logged : bool
        ``True`` when explanation metrics were logged for the current trial.
    exp_accuracy : dict[str, float]
        Dictionary containing explanation accuracy metrics with keys
        ``"auroc"``, ``"auprc"``, and ``"f1"``.
    ans_accuracy_val : float
        Answer accuracy for the trial.
    pred_edge_mask : torch.Tensor | None
        Predicted edge mask from the explanation, or ``None`` if no explanation
        was generated.
    """
    model_device = wrapper.model.device
    pyg_batch = create_pyg_batch(sample["graph"], device=model_device)
    explanation, ans_accuracy = _generate_explanation(
        wrapper, sample, pyg_batch, gen_cfg, explainer_args=explainer_args, num_trials=num_gen_trials
    )

    if explanation is None:
        exp_accuracy = {"auroc": 0.0, "auprc": 0.0, "f1": 0.0}
        ans_accuracy_val = 0.0
        return False, exp_accuracy, ans_accuracy_val, None

    # Compute explanation accuracy
    gt_edge_mask = _get_gt_explanation(sample)
    pred_edge_mask = explanation.edge_mask.detach().cpu().float()
    auroc, f1 = groundtruth_metrics(pred_edge_mask, gt_edge_mask, metrics=["auroc", "f1_score"])
    auprc = average_precision(pred_edge_mask, gt_edge_mask.int(), task="binary").item()
    exp_accuracy = {"auroc": float(auroc), "auprc": float(auprc), "f1": float(f1)}
    ans_accuracy_val = float(ans_accuracy)

    # Log explanation accuracy for positive samples
    metrics_logged = False
    if dataset_name == "MotifQA" and len(sample.get("motif_nodes", [])) > 0:
        record = {
            "sample_index": sample["index"],
            "trial": trial_idx,
            "answer_accuracy": ans_accuracy_val,
            **exp_accuracy,
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
            exp_accuracy=exp_accuracy,
            ans_accuracy=ans_accuracy_val,
        )
    else:
        explanation.visualize_graph(graph_path)
    feature_path = os.path.join(node_feat_dir, f"node_feat_{suffix}.svg")
    explanation.visualize_feature_importance(feature_path)

    return metrics_logged, exp_accuracy, ans_accuracy_val, pred_edge_mask


def process_dataset(
    dataset: Iterable[dict[str, str]],
    args: argparse.Namespace,
    log_path: str,
    fieldnames: list[str],
    show_progress: bool,
    model: GraphTokenLM,
    tokenizer: AutoTokenizer,
) -> tuple[float, float, float, float, int, dict[int, list[torch.Tensor]], dict[int, dict[str, float]]]:
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
    gen_cfg = GenerationConfig(
        max_new_tokens=10,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    explainer_args: dict[str, float | int] = {
        "epochs": args.epochs,
        "lr": args.lr,
        "edge_size": args.edge_size,
        "edge_ent": args.edge_ent,
    }

    total_f1, total_auroc, total_auprc, total_answer_accuracy = 0.0, 0.0, 0.0, 0.0
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

    has_trial_override = False
    if hasattr(dataset, "column_names") and TRIAL_OVERRIDE_COLUMN in dataset.column_names:
        # Ensure at least one sample carries an override before switching modes.
        if len(dataset) > 0 and dataset[0].get(TRIAL_OVERRIDE_COLUMN) is not None:
            has_trial_override = True

    per_sample_trials = 1 if has_trial_override else args.num_trials
    total_steps = len(dataset) * per_sample_trials
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
            logged, exp_accuracy, ans_accuracy_single, edge_mask = explain_sample(
                wrapper=wrapper,
                sample=sample,
                trial_idx=i,
                num_trials=args.num_trials,
                gen_cfg=gen_cfg,
                log_path=log_path,
                fieldnames=fieldnames,
                dataset_name=args.dataset,
                explainer_args=explainer_args,
            )
            if edge_mask is not None:
                sample_edge_masks[sample_idx].append(edge_mask)
            if logged:
                total_auroc += exp_accuracy["auroc"]
                total_auprc += exp_accuracy["auprc"]
                total_f1 += exp_accuracy["f1"]
                total_answer_accuracy += ans_accuracy_single
                total_count += 1
                stats = sample_metrics[sample_idx]
                stats["answer_accuracy_sum"] += ans_accuracy_single
                stats["auroc_sum"] += exp_accuracy["auroc"]
                stats["auprc_sum"] += exp_accuracy["auprc"]
                stats["f1_sum"] += exp_accuracy["f1"]
                stats["count"] += 1
            if progress is not None:
                if args.wandb:
                    wandb.log({"explanation/steps": progress.n})
                progress.update(1)

    if progress is not None:
        progress.close()

    return (
        total_auroc,
        total_auprc,
        total_f1,
        total_answer_accuracy,
        total_count,
        dict(sample_edge_masks),
        {idx: dict(stats) for idx, stats in sample_metrics.items()},
    )


def main():
    set_seed(42)
    args = build_args()
    run_name = datetime.now().strftime("%m%d-%H%M")

    # Setup DDP, random seed, and device
    rank, world_size, local_rank, is_distributed = _init_distributed_if_needed()
    set_seed(42 + rank)

    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_rank0 = rank == 0

    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path, device=device, verbose=is_rank0)
    model.eval()

    # Load dataset and apply filtering
    dataset = build_dataset(
        args.dataset,
        args.subset,
        args.split,
        node_feat_dim=model.config.node_feat_dim,
    )
    dataset, OUT_DIR = filter_dataset(dataset, args, run_name)
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

    # Setup wandb logging (rank 0 only)
    if args.wandb and is_rank0:
        wandb.init(project="MotifQA-Explainer", name=run_name, dir=OUT_DIR)

    # Setup CSV logging
    os.makedirs(OUT_DIR, exist_ok=True)
    base_log_path = os.path.join(OUT_DIR, "sample_metrics.csv")
    shard_log_path = base_log_path if world_size == 1 else os.path.join(OUT_DIR, f"metrics_rank{rank}.csv")

    fieldnames = ["sample_index", "trial", "answer_accuracy", "auroc", "auprc", "f1"]
    _write_metrics_header(shard_log_path, fieldnames)

    # Process dataset and collect metrics
    total_auroc, total_auprc, total_f1, total_answer_accuracy, total_count, sample_edge_masks, sample_metrics = (
        process_dataset(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            log_path=shard_log_path,
            fieldnames=fieldnames,
            show_progress=(is_rank0 and len(dataset) > 0),
            args=args,
        )
    )

    # Aggregate metrics across ranks
    metrics_tensor = torch.tensor(
        [total_f1, total_auroc, total_auprc, total_answer_accuracy, float(total_count)],
        device=device,
    )
    if is_distributed:
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    total_f1, total_auroc, total_auprc, total_answer_accuracy, total_count = metrics_tensor.tolist()
    total_count = int(total_count)

    # Merge per-trial metrics CSVs across ranks
    if is_distributed:
        barrier_kwargs: dict[str, object] = {}
        if device.type == "cuda" and device.index is not None:
            barrier_kwargs["device_ids"] = [device.index]
        dist.barrier(**barrier_kwargs)
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
        dist.barrier(**barrier_kwargs)

    # Merge per-trial edge masks and metrics across ranks
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

    # Compute and record average metrics across all positive samples
    if is_rank0:
        if total_count > 0:
            avg_answer_accuracy = total_answer_accuracy / total_count
            avg_f1 = total_f1 / total_count
            avg_auroc = total_auroc / total_count
            avg_auprc = total_auprc / total_count
            print(
                "Average explanation accuracy across positive samples: "
                f"AnswerAcc={avg_answer_accuracy:.3f}, "
                f"AUROC={avg_auroc:.3f}, AUPRC={avg_auprc:.3f}, F1={avg_f1:.3f}"
            )
            print(f"Saved explanation metrics to {base_log_path}")
        else:
            avg_answer_accuracy = 0.0
            print("No explanation metrics recorded for positive samples.")

        # Save average metrics per sample
        avg_metrics_path = os.path.join(OUT_DIR, "average_metrics.csv")
        write_average_metrics_csv(
            avg_metrics_path,
            merged_sample_metrics,
            merged_edge_masks,
        )
        print(f"[INFO] Saved average metrics to {avg_metrics_path}")
        if args.wandb and total_count > 0:
            wandb.log(
                {
                    "avg_answer_accuracy": avg_answer_accuracy,
                    "avg_f1": avg_f1,
                    "avg_auroc": avg_auroc,
                    "avg_auprc": avg_auprc,
                }
            )

    if args.wandb:
        wandb.finish()

    if is_distributed:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
