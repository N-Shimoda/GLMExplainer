import argparse
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Literal

import torch
import torch.distributed as dist
import wandb
from torch_geometric.explain import Explainer, GNNExplainer, groundtruth_metrics
from torchmetrics.functional import average_precision
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig
from transformers.trainer_utils import set_seed

from eval import MAX_NEW_TOKENS, create_pyg_batch
from src.ckpt import _resolve_model_path
from src.constants import GRAPHQA_SUBSETS, MOTIFQA_SUBSETS
from src.explanation.args import check_non_negative_int, validate_args
from src.explanation.logging import (
    _record_sample_average_metrics,
    _write_metrics_header,
    append_run_history_row,
    write_average_metrics_csv,
)
from src.explanation.metrics import EDGE_MASK_STABILITY_KEYS
from src.explanation.preprocess import build_dataset, filter_dataset
from src.explanation.wrapper import GLMWrapper
from src.glm import GraphTokenLM
from src.metrics import comp_accuracy
from src.utils import visualize_motif_explanation

GRAPH_PDF_SUBDIR = "graphs"
IMPORTANCE_SUBDIR = "importance"
OUTPUT_TEXTS_FILENAME = "output_texts.jsonl"
DATASET_FILENAME = "filtered_dataset.jsonl"
TRIAL_OVERRIDE_COLUMN = "_trial_override"
AVERAGE_METRIC_FIELDNAMES = [
    "sample_index",
    "answer_accuracy",
    "auroc",
    "auprc",
    "f1",
    *EDGE_MASK_STABILITY_KEYS,
    "edge_mask_size",
    "edge_mask_ent",
]


@dataclass(frozen=True)
class ExplainSampleConfig:
    """Configuration bundle for explain_sample to keep the call site compact."""

    num_trials: int
    num_gen_trials: int
    min_correct_answers: int
    explainer_args: dict[str, float | int]
    llr_threshold: float | None
    f1_threshold: float
    baseline_graph: str
    dataset_name: str
    subset: str
    log_path: str
    fieldnames: list[str]
    file_type: Literal["svg", "pdf"]


def _init_distributed_if_needed() -> tuple[int, int, int, bool]:
    """Initialize torch.distributed and return rank metadata if WORLD_SIZE > 1.

    Returns
    -------
    rank : int
        The global rank of the current process.
    world_size : int
        The total number of processes in the distributed setup.
    local_rank : int
        The local rank of the current process on its node.
    is_distributed : bool
        ``True`` if distributed training was initialized, ``False`` otherwise.
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

    init_kwargs: dict[str, object] = {"backend": backend, "timeout": timedelta(hours=1)}
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
    p = argparse.ArgumentParser(description="Explain GraphTokenLM predictions using GNNExplainer")

    # Model checkpoint
    p.add_argument("--model-path", type=str, required=True, help="Path to the model checkpoint")

    # Dataset setting
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
        choices=MOTIFQA_SUBSETS + GRAPHQA_SUBSETS,
        help="Dataset subset to use. Only applicable for GraphQA.",
    )
    p.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        default="test",
        help="Dataset split to use (default: test)",
    )

    # Sample filtering
    p.add_argument(
        "--target-pos-samples",
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
    p.add_argument(
        "--num-gen-trials", type=int, default=10, help="Number of generation trials per explanation (default: 10)"
    )
    p.add_argument(
        "--min-correct-answers",
        type=check_non_negative_int,
        default=0,
        help=(
            "Minimum number of correct generations required (exclusive) before running the explainer. "
            "Explanations run only if correct_count > min_correct_answers (default: 0)."
        ),
    )

    # Hyper-parameters for GNNExplainer
    p.add_argument("--edge-size", type=float, default=0.005, help="GNNExplainer edge size parameter (default: 0.005)")
    p.add_argument("--edge-ent", type=float, default=1.0, help="GNNExplainer edge entropy parameter (default: 1.0)")
    p.add_argument("--epochs", type=int, default=200, help="GNNExplainer optimization epochs (default: 200)")
    p.add_argument("--lr", type=float, default=0.01, help="GNNExplainer learning rate (default: 0.01)")

    # Relevant token selection
    p.add_argument(
        "--llr-threshold",
        type=float,
        default=None,
        help=(
            "Log-likelihood ratio threshold for selecting relevant tokens. "
            "Only tokens with LLR above this value will be included in the explanation."
        ),
    )
    p.add_argument(
        "--baseline-graph",
        type=str,
        default=None,
        choices=["complete", "empty"],
        help="Baseline graph type for LLR computation.",
    )

    # Explanation metrics
    p.add_argument(
        "--f1-threshold",
        type=float,
        default=0.5,
        help="Threshold for computing F1 score against ground-truth explanations (default: 0.5)",
    )

    # Logging
    p.add_argument(
        "--outdir-base", type=str, default="explanations", help="Base path of output directory (default: explanations)"
    )
    p.add_argument(
        "--output-file-type",
        type=str,
        default="svg",
        choices=["svg", "pdf"],
        help="File type for saved figures (default: svg)",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Log per-sample explanation metrics to Weights & Biases.",
    )
    p.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Optional W&B tags (space-separated).",
    )

    # Parse and validate args
    args = p.parse_args()
    validate_args(args)

    jaccard_k_by_subset = {
        "ba_shapes": 6,
        "tree_cycle": 6,
        "tree_grid": 12,
        "tree_grid_v2": 7,
        "ba_two_motifs": 5,
        "shortest_path": 6,
    }
    args.jaccard_k = jaccard_k_by_subset.get(args.subset, 6)

    # Extract explainer args
    explainer_keys = ["epochs", "lr", "edge_size", "edge_ent"]
    explainer_args = {key: getattr(args, key) for key in explainer_keys}
    for key in explainer_keys:
        delattr(args, key)

    return args, explainer_args


def load_model(
    model_path: str, device: torch.device | str, verbose: bool = True
) -> tuple[GraphTokenLM, AutoTokenizer]:
    """Loads the GraphTokenLM model and tokenizer from a local checkpoint or HF Hub model.

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
    ckpt_path : str
        Resolved checkpoint path from which the model was loaded.
    """
    # Resolve local checkpoint path or HF Hub identifier
    resolved_model_path, run_name = _resolve_model_path(model_path)

    # Load model onto the specified device
    model = GraphTokenLM.from_pretrained(resolved_model_path, load_llm_weights=False)
    model.to(torch.device(device))
    if verbose:
        print(f"Loaded model from {resolved_model_path} (run name: {run_name})")

    # Load pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model.config.llm_name, trust_remote_code=True)
    return model, tokenizer, resolved_model_path


def _get_gt_explanation(sample: dict[str, str]) -> torch.Tensor:
    """
    Build the binary ground-truth edge mask for a MotifQA sample.

    Parameters
    ----------
    sample : dict[str, str]
        Dataset sample containing at least ``graph``, ``nodes``, ``edges``, and ``motif_nodes`` fields.
        ``graph`` must hold a PyG-compatible dictionary whose ``edge_index`` encodes the bidirectional edge list.

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


def explain_sample(
    wrapper: GLMWrapper,
    sample: dict[str, str],
    trial_idx: int,
    gen_cfg: GenerationConfig,
    cfg: ExplainSampleConfig,
) -> tuple[bool, dict[str, float], float, torch.Tensor | None]:
    """Explain a single dataset sample, collect metrics, and persist trial artifacts.

    Parameters
    ----------
    wrapper : GLMWrapper
        Wrapper around the GraphToken language model that exposes convenience
        methods for generation and explanation.
    sample : dict[str, str]
        Dataset entry that must contain the graph structure as well as fields
        required by :func:`create_pyg_batch`.
    trial_idx : int
        Index of the current trial for the given ``sample``.
    gen_cfg : GenerationConfig
        Configuration controlling the language-model generation step.
    cfg : ExplainSampleConfig
        Configuration bundle for explanation and logging settings.

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

    # Generate output texts
    output_texts = wrapper.gen_output(
        input_text=sample["prompt"], graph=pyg_batch, gen_cfg=gen_cfg, num_trials=cfg.num_gen_trials
    )
    record = {
        "sample_index": sample["index"],
        "output_texts": output_texts,
    }

    # Append output texts to JSONL log
    output_texts_path = os.path.join(os.path.dirname(cfg.log_path), OUTPUT_TEXTS_FILENAME)
    with open(output_texts_path, "a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=True) + "\n")

    # Compute answer accuracy to find the first correct generation
    ans_accuracy, _, correct_mask = comp_accuracy(output_texts, [sample["completion"]] * len(output_texts), cfg.subset)
    correct_count = sum(correct_mask)
    if correct_count <= cfg.min_correct_answers:
        print(
            f"[WARN] Sample[index={sample['index']}] has {correct_count} correct answers "
            f"(threshold: >{cfg.min_correct_answers}); skipping explanation."
        )
        exp_accuracy = {"auroc": 0.0, "auprc": 0.0, "f1": 0.0}
        ans_accuracy_val = 0.0
        return False, exp_accuracy, ans_accuracy_val, None
    first_correct_idx = correct_mask.index(True)
    wrapper.set_generated_ids(output_texts[first_correct_idx])

    # Set relevant token IDs if LLR threshold is specified
    wrapper.relevant_idx = None
    if cfg.llr_threshold is not None and cfg.llr_threshold > 0.0:
        wrapper.set_relevant_ids(cfg.baseline_graph, llr_threshold=cfg.llr_threshold)

    # Generate explanation using GNNExplainer
    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(num_hops=wrapper.model.config.num_gnn_layers, **cfg.explainer_args),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="graph",
            return_type="raw",
        ),
    )
    explanation = explainer(x=pyg_batch.x, edge_index=pyg_batch.edge_index, batch=pyg_batch.batch)

    # Compute explanation accuracy
    gt_edge_mask = _get_gt_explanation(sample)
    pred_edge_mask = explanation.edge_mask.detach().cpu().float()
    edge_mask_metrics = _compute_edge_mask_metrics(pred_edge_mask)
    auroc, f1 = groundtruth_metrics(
        pred_edge_mask,
        gt_edge_mask,
        metrics=["auroc", "f1_score"],
        threshold=cfg.f1_threshold,
    )
    auprc = average_precision(pred_edge_mask, gt_edge_mask.int(), task="binary").item()
    exp_accuracy = {"auroc": float(auroc), "auprc": float(auprc), "f1": float(f1)}
    ans_accuracy_val = float(ans_accuracy)

    # Log explanation accuracy for positive samples
    metrics_logged = False
    if cfg.dataset_name == "MotifQA" and len(sample.get("motif_nodes", [])) > 0:
        record = {
            "sample_index": sample["index"],
            "trial": trial_idx,
            "answer_accuracy": ans_accuracy_val,
            **exp_accuracy,
            **edge_mask_metrics,
        }
        with open(cfg.log_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=cfg.fieldnames)
            writer.writerow(record)
        metrics_logged = True

    # Directories to save figures and raw edge importance
    suffix = f"{sample['index']}_{trial_idx}" if cfg.num_trials > 1 else f"{sample['index']}"
    out_dir = os.path.dirname(cfg.log_path)
    graph_dir = os.path.join(out_dir, GRAPH_PDF_SUBDIR, f"graph_{sample['index']}")
    importance_dir = os.path.join(out_dir, IMPORTANCE_SUBDIR, f"graph_{sample['index']}")
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(importance_dir, exist_ok=True)

    # Save visualizations
    graph_path = os.path.join(graph_dir, f"{suffix}.{cfg.file_type}")
    if cfg.dataset_name == "MotifQA":
        visualize_motif_explanation(
            sample=sample,
            explanation=explanation,
            graph_path=graph_path,
            exp_accuracy=exp_accuracy,
            ans_accuracy=ans_accuracy_val,
        )
    else:
        explanation.visualize_graph(graph_path)

    # Save raw edge importance values with edge index mapping
    importance_path = os.path.join(importance_dir, f"{suffix}.csv")
    edge_index = pyg_batch.edge_index.detach().cpu()
    with open(importance_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["src", "dst", "edge_mask"])
        for (src, dst), score in zip(edge_index.t().tolist(), pred_edge_mask.tolist()):
            writer.writerow([src, dst, score])

    return metrics_logged, exp_accuracy, ans_accuracy_val, pred_edge_mask


def _compute_edge_mask_metrics(edge_mask: torch.Tensor | None) -> dict[str, float]:
    """Compute edge mask summary metrics (sum and entropy)."""
    if edge_mask is None or edge_mask.numel() == 0:
        return {"edge_mask_size": 0.0, "edge_mask_ent": 0.0}
    mask = edge_mask.float().clamp(min=0.0)
    total = float(mask.sum().item())
    if total <= 0.0:
        return {"edge_mask_size": total, "edge_mask_ent": 0.0}
    probs = mask / total
    entropy = float(-(probs * (probs + 1e-12).log()).sum().item())
    return {"edge_mask_size": total, "edge_mask_ent": entropy}


def process_dataset(
    dataset: Iterable[dict[str, str]],
    model: GraphTokenLM,
    tokenizer: AutoTokenizer,
    subset: str,
    log_path: str,
    fieldnames: list[str],
    show_progress: bool,
    args: argparse.Namespace,
    explainer_args: dict[str, float | int],
    avg_log_path: str | None = None,
    avg_fieldnames: list[str] | None = None,
) -> tuple[dict[str, float], float, int, dict[int, list[torch.Tensor]], dict[int, dict[str, float]]]:
    """Run explanations across the dataset, tracking trial artifacts and aggregates.

    Parameters
    ----------
    dataset : Iterable[dict[str, str]]
        Iterable of dataset samples. Each sample must at least expose the
        ``index`` key and any fields required by :func:`explain_sample`,
        including ``graph`` and model inputs.
    model : GraphTokenLM
        Pretrained GraphToken language model whose predictions are explained.
    tokenizer : AutoTokenizer
        Tokenizer paired with ``model`` and used to build prompts.
    subset : str
        Name of the dataset subset being processed (e.g., ``"ba_shapes"``).
    log_path : str
        CSV path forwarded to :func:`explain_sample` for appending per-trial
        metrics.
    fieldnames : list[str]
        Ordered column names used by the CSV logger.
    show_progress : bool
        If ``True``, render a ``tqdm`` progress bar while processing samples.
    args : argparse.Namespace
        Parsed CLI arguments controlling trial repetition, dataset name, and
        optional per-sample overrides such as ``num_trials`` and ``dataset``.
    explainer_args : dict[str, float | int]
        Hyperparameters passed to the explainer, e.g., epochs and learning rate.
    avg_log_path : str, optional
        CSV destination used to record per-sample averaged metrics once a sample
        completes its configured ``num_trials``.
    avg_fieldnames : list[str], optional
        Column ordering applied when writing per-sample averages.

    Returns
    -------
    exp_metric_totals : dict[str, float]
        Running sums of AUROC, AUPRC, and F1 computed over the logged trials.
    total_answer_accuracy : float
        Sum of answer accuracies recorded alongside the explanation metrics.
    total_count : int
        Number of trials that yielded metrics (i.e., were appended to the CSV).
    sample_edge_masks : dict[int, list[torch.Tensor]]
        Mapping from each sample index to the edge masks produced across its trials.
    sample_metrics : dict[int, dict[str, float]]
        Per-sample aggregates containing running sums for AUROC, AUPRC, F1, and
        answer accuracy along with a ``count`` describing how many trials contributed.

    Notes
    -----
    If the dataset length is unknown, it is materialized into a list to enable
    progress reporting. When the dataset carries a ``_trial_override`` column,
    those overrides supersede ``args.num_trials`` for the affected samples.
    """
    # Initialize model wrapper
    wrapper = GLMWrapper(model, tokenizer)
    gen_cfg = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS[subset],
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    sample_cfg = ExplainSampleConfig(
        num_trials=args.num_trials,
        num_gen_trials=args.num_gen_trials,
        min_correct_answers=args.min_correct_answers,
        explainer_args=explainer_args,
        llr_threshold=args.llr_threshold,
        f1_threshold=args.f1_threshold,
        baseline_graph=args.baseline_graph,
        dataset_name=args.dataset,
        subset=subset,
        log_path=log_path,
        fieldnames=fieldnames,
        file_type=args.output_file_type,
    )

    # Prepare logging
    exp_metric_totals = {"auroc": 0.0, "auprc": 0.0, "f1": 0.0}
    total_answer_accuracy = 0.0
    total_count = 0
    sample_edge_masks: defaultdict[int, list[torch.Tensor]] = defaultdict(list)
    sample_metrics: defaultdict[int, dict[str, float]] = defaultdict(
        lambda: {
            "answer_accuracy_sum": 0.0,
            "auroc_sum": 0.0,
            "auprc_sum": 0.0,
            "f1_sum": 0.0,
            "edge_mask_size_sum": 0.0,
            "edge_mask_ent_sum": 0.0,
            "count": 0,
        }
    )
    trial_completion_counts: defaultdict[int, int] = defaultdict(int)
    finalized_samples: set[int] = set()

    # Check for per-sample trial overrides
    has_trial_override = False
    if hasattr(dataset, "column_names") and TRIAL_OVERRIDE_COLUMN in dataset.column_names:
        # Ensure at least one sample carries an override before switching modes.
        if len(dataset) > 0 and dataset[0].get(TRIAL_OVERRIDE_COLUMN) is not None:
            has_trial_override = True

    # Setup progress bar
    per_sample_trials = 1 if has_trial_override else args.num_trials
    total_steps = len(dataset) * per_sample_trials
    progress = tqdm(total=total_steps) if show_progress and total_steps > 0 else None

    start_time = time.time()

    for sample in dataset:
        sample_idx = sample["index"]
        override_value = sample.get(TRIAL_OVERRIDE_COLUMN) if has_trial_override else None
        if override_value is not None:
            trial_indices = [int(override_value)]
        else:
            trial_indices = range(args.num_trials)

        num_trial_runs = len(trial_indices)
        for i in trial_indices:
            logged, exp_accuracy, ans_accuracy_single, edge_mask = explain_sample(
                wrapper=wrapper,
                sample=sample,
                trial_idx=i,
                gen_cfg=gen_cfg,
                cfg=sample_cfg,
            )
            if edge_mask is not None:
                sample_edge_masks[sample_idx].append(edge_mask)
            if logged:
                edge_mask_metrics = _compute_edge_mask_metrics(edge_mask)
                exp_metric_totals["auroc"] += exp_accuracy["auroc"]
                exp_metric_totals["auprc"] += exp_accuracy["auprc"]
                exp_metric_totals["f1"] += exp_accuracy["f1"]
                total_answer_accuracy += ans_accuracy_single
                total_count += 1
                stats = sample_metrics[sample_idx]
                stats["answer_accuracy_sum"] += ans_accuracy_single
                stats["auroc_sum"] += exp_accuracy["auroc"]
                stats["auprc_sum"] += exp_accuracy["auprc"]
                stats["f1_sum"] += exp_accuracy["f1"]
                stats["edge_mask_size_sum"] += edge_mask_metrics["edge_mask_size"]
                stats["edge_mask_ent_sum"] += edge_mask_metrics["edge_mask_ent"]
                stats["count"] += 1
            if progress is not None:
                if args.wandb:
                    elapsed = time.time() - start_time
                    wandb.log(
                        {
                            "explain/steps": progress.n,
                            "explain/seconds_per_step": elapsed / (progress.n + 1),
                            "explain/elapsed_time": elapsed,
                        }
                    )
                progress.update(1)

        trial_completion_counts[sample_idx] += num_trial_runs
        if sample_idx not in finalized_samples and trial_completion_counts[sample_idx] >= args.num_trials:
            _record_sample_average_metrics(
                avg_log_path=avg_log_path,
                fieldnames=avg_fieldnames,
                sample_idx=sample_idx,
                stats=sample_metrics.get(sample_idx),
                edge_masks=sample_edge_masks.get(sample_idx, []),
                jaccard_k=args.jaccard_k,
            )
            finalized_samples.add(sample_idx)

    if progress is not None:
        progress.close()

    return (
        dict(exp_metric_totals),
        total_answer_accuracy,
        total_count,
        dict(sample_edge_masks),
        {idx: dict(stats) for idx, stats in sample_metrics.items()},
    )


def main():
    """Compute edge importance explanations for GraphTokenLM predictions on specified dataset samples."""
    args, explainer_args = build_args()
    date_str = datetime.now().strftime("%m%d-%H%M")
    run_name = f"{args.subset}_{date_str}"
    OUT_DIR = os.path.join(args.outdir_base, args.subset, date_str)
    os.makedirs(OUT_DIR, exist_ok=True)

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
    model, tokenizer, ckpt_path = load_model(args.model_path, device=device, verbose=is_rank0)
    model.eval()

    # Load dataset and apply filtering
    lpe_dim = getattr(model.config, "lpe_dim", model.config.node_feat_dim)
    use_degree_emb = getattr(model.config, "use_degree_emb", False)
    dataset = build_dataset(
        args.dataset,
        args.subset,
        args.split,
        lpe_dim=lpe_dim,
        use_degree_emb=use_degree_emb,
    )
    dataset = filter_dataset(
        dataset,
        dataset_name=args.dataset,
        sample_idx=args.sample_idx,
        target_pos_samples=args.target_pos_samples,
        target_value=args.target_value,
        num_samples=args.num_samples,
    )

    if is_rank0:
        debug_dataset_path = os.path.join(OUT_DIR, DATASET_FILENAME)
        dataset.to_json(debug_dataset_path, lines=True)

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
    args.model_path = ckpt_path
    wandb_tags = args.tags
    delattr(args, "tags")
    if args.wandb and is_rank0:
        wandb.init(
            project="MotifQA-Explainer",
            name=run_name,
            config={**vars(args), **explainer_args, "lpe_dim": lpe_dim, "use_degree_emb": use_degree_emb},
            tags=wandb_tags,
            dir=OUT_DIR,
        )

    # Setup CSV logging
    os.makedirs(OUT_DIR, exist_ok=True)
    base_log_path = os.path.join(OUT_DIR, "sample_metrics.csv")
    shard_log_path = base_log_path if world_size == 1 else os.path.join(OUT_DIR, f"metrics_rank{rank}.csv")

    fieldnames = [
        "sample_index",
        "trial",
        "answer_accuracy",
        "auroc",
        "auprc",
        "f1",
        "edge_mask_size",
        "edge_mask_ent",
    ]
    _write_metrics_header(shard_log_path, fieldnames)
    avg_metrics_base_path = os.path.join(OUT_DIR, "average_metrics.csv")
    avg_metrics_shard_path = (
        avg_metrics_base_path if world_size == 1 else os.path.join(OUT_DIR, f"average_metrics_rank{rank}.csv")
    )
    with open(avg_metrics_shard_path, "w", newline="") as avg_file:
        writer = csv.DictWriter(avg_file, fieldnames=AVERAGE_METRIC_FIELDNAMES)
        writer.writeheader()

    # Process dataset and collect metrics
    exp_metric_totals, total_answer_accuracy, total_count, sample_edge_masks, sample_metrics = process_dataset(
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        subset=args.subset,
        log_path=shard_log_path,
        fieldnames=fieldnames,
        show_progress=(is_rank0 and len(dataset) > 0),
        args=args,
        explainer_args=explainer_args,
        avg_log_path=avg_metrics_shard_path,
        avg_fieldnames=AVERAGE_METRIC_FIELDNAMES,
    )

    # Aggregate metrics across ranks
    metrics_tensor = torch.tensor(
        [
            exp_metric_totals["auroc"],
            exp_metric_totals["auprc"],
            exp_metric_totals["f1"],
            total_answer_accuracy,
            float(total_count),
        ],
        device=device,
    )
    if is_distributed:
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    auroc_sum, auprc_sum, f1_sum, total_answer_accuracy, total_count = metrics_tensor.tolist()
    exp_metric_totals = {"auroc": auroc_sum, "auprc": auprc_sum, "f1": f1_sum}
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
                    "auroc_sum": 0.0,
                    "auprc_sum": 0.0,
                    "f1_sum": 0.0,
                    "edge_mask_size_sum": 0.0,
                    "edge_mask_ent_sum": 0.0,
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
                    acc["auroc_sum"] += stats.get("auroc_sum", 0.0)
                    acc["auprc_sum"] += stats.get("auprc_sum", 0.0)
                    acc["f1_sum"] += stats.get("f1_sum", 0.0)
                    acc["edge_mask_size_sum"] += stats.get("edge_mask_size_sum", 0.0)
                    acc["edge_mask_ent_sum"] += stats.get("edge_mask_ent_sum", 0.0)
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
        avg_answer_accuracy = 0.0
        avg_auroc = 0.0
        avg_auprc = 0.0
        avg_f1 = 0.0
        avg_edge_size = 0.0
        avg_edge_ent = 0.0
        samples_used_count = 0
        samples_total_count = 0
        samples_used_pct = 0.0
        if merged_sample_metrics is not None:
            samples_total_count = len(merged_sample_metrics)
            samples_used_count = sum(1 for stats in merged_sample_metrics.values() if stats.get("count", 0) > 0)
            if samples_total_count > 0:
                samples_used_pct = (samples_used_count / samples_total_count) * 100.0
        if total_count > 0:
            avg_answer_accuracy = total_answer_accuracy / total_count
            avg_auroc = exp_metric_totals["auroc"] / total_count
            avg_auprc = exp_metric_totals["auprc"] / total_count
            avg_f1 = exp_metric_totals["f1"] / total_count
            if merged_sample_metrics is not None:
                total_edge_size = sum(stats.get("edge_mask_size_sum", 0.0) for stats in merged_sample_metrics.values())
                total_edge_ent = sum(stats.get("edge_mask_ent_sum", 0.0) for stats in merged_sample_metrics.values())
                avg_edge_size = total_edge_size / total_count
                avg_edge_ent = total_edge_ent / total_count
            print(
                "[INFO] Metrics across positive samples: "
                f"AnswerAcc={avg_answer_accuracy:.3f}, "
                f"AUROC={avg_auroc:.3f}, AUPRC={avg_auprc:.3f}, F1={avg_f1:.3f}"
            )
            print(f"[INFO] Saved explanation metrics to {base_log_path}")
        else:
            print("[WARN] No explanation metrics recorded for positive samples.")

        # Save average metrics per sample
        avg_metrics_path = os.path.join(OUT_DIR, "average_metrics.csv")
        _, stability_metrics = write_average_metrics_csv(
            avg_metrics_path,
            merged_sample_metrics,
            merged_edge_masks,
            jaccard_k=args.jaccard_k,
        )
        print(f"[INFO] Saved average metrics to {avg_metrics_path}")
        if is_distributed:
            for idx in range(world_size):
                rank_avg_path = os.path.join(OUT_DIR, f"average_metrics_rank{idx}.csv")
                if os.path.exists(rank_avg_path):
                    os.remove(rank_avg_path)

        history_path = os.path.join(os.path.dirname(OUT_DIR), "run_history.csv")
        append_run_history_row(
            history_path,
            {
                "run_name": date_str,
                "avg_answer_accuracy": avg_answer_accuracy,
                "avg_auroc": avg_auroc,
                "avg_auprc": avg_auprc,
                "avg_f1": avg_f1,
                "avg_edge_size": avg_edge_size,
                "avg_edge_ent": avg_edge_ent,
                **stability_metrics,
                **explainer_args,
                "llr_threshold": args.llr_threshold,
            },
        )

        if args.wandb:
            wandb_payload = stability_metrics.copy()
            wandb_payload["samples_used_pct"] = samples_used_pct
            wandb_payload["samples_used_count"] = samples_used_count
            wandb_payload["avg_edge_size"] = avg_edge_size
            wandb_payload["avg_edge_ent"] = avg_edge_ent
            if total_count > 0:
                wandb_payload.update(
                    {
                        "avg_answer_accuracy": avg_answer_accuracy,
                        "avg_auroc": avg_auroc,
                        "avg_auprc": avg_auprc,
                        "avg_f1": avg_f1,
                    }
                )
            wandb.log(wandb_payload)

    if args.wandb:
        wandb.finish()

    if is_distributed:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
