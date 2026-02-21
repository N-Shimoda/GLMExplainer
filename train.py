import argparse
import os
import shutil
from datetime import datetime
from math import ceil

import datasets
import torch
import torch.distributed as dist
import wandb
from datasets import concatenate_datasets, load_dataset
from datasets.arrow_dataset import Dataset
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed
from trl import SFTConfig, SFTTrainer

from eval import EXT_MAX_NEW_TOKENS, MAX_NEW_TOKENS, collect_result, eval_model
from src.collator import GraphQACollator
from src.constants import GRAPHQA_SUBSETS, MOTIFQA_SUBSETS
from src.glm import VALID_GRAPH_POOLING, GraphTokenLM, GraphTokenLMConfig
from src.preprocess import add_graph_column


def is_main_process() -> bool:
    """Check if the current process is the main process in DDP setup."""
    return int(os.environ.get("RANK", "0")) == 0


def _safe_barrier():
    """Call dist.barrier with device_ids when using NCCL to silence warnings."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def validate_args(args: argparse.Namespace):
    if not hasattr(args, "dataset") or not hasattr(args, "subset"):
        # Skip validation for multitask helper scripts that don't define dataset/subset.
        return

    # Subsets
    subsets = args.subset
    match args.dataset:
        case "GraphQA":
            valid_subsets = GRAPHQA_SUBSETS
        case "MotifQA":
            valid_subsets = MOTIFQA_SUBSETS
    invalid = [subset for subset in subsets if subset not in valid_subsets]
    if invalid:
        raise ValueError(f"Subsets {invalid} are not valid for dataset {args.dataset}.")

    # Custom dataset
    if args.use_custom_dataset and args.dataset != "GraphQA":
        raise ValueError("--use-custom-dataset is only supported with GraphQA dataset.")
    if args.use_custom_dataset and len(subsets) != 1:
        raise ValueError("--use-custom-dataset only supports a single subset.")

    # Checkpointing
    if args.no_save and args.save_intermediate_models:
        raise ValueError("--no-save and --save-intermediate-models cannot be used together.")
    if args.no_save and not args.wandb:
        raise ValueError("--no-save without --wandb is prohibited since no checkpoints are saved locally.")


def build_args():
    p = argparse.ArgumentParser(description="Train GraphTokenLM on GraphQA or MotifQA dataset.")

    # Dataset
    p.add_argument("--dataset", type=str, default="GraphQA", choices=["GraphQA", "MotifQA"])
    p.add_argument(
        "--subset",
        type=str,
        nargs="+",
        required=True,
        choices=GRAPHQA_SUBSETS + MOTIFQA_SUBSETS,
        help="One or more subsets. Passing multiple subsets enables multitask training.",
    )
    p.add_argument("--use-custom-dataset", action="store_true", help="Use custom dataset with extended answer labels.")

    # Model architecture
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Base")
    p.add_argument(
        "--gnn-type",
        type=str,
        default="GCN",
        choices=["GCN", "GAT", "GIN", "GraphSAGE", "GraphTransformer"],
    )
    p.add_argument("--num-max-nodes", type=int, default=20)
    p.add_argument("--num-graph-tokens", type=int, default=4)
    p.add_argument(
        "--graph-pooling",
        type=str,
        nargs="+",
        default=["mean"],
        choices=VALID_GRAPH_POOLING,
        help="Pooling strategy to aggregate node embeddings into graph embeddings. Pass one to three values.",
    )
    p.add_argument("--pos-emb-dim", type=int, default=8)
    p.add_argument("--lpe-dim", type=int, default=8)
    p.add_argument("--use-degree-emb", action="store_true")
    p.add_argument("--gnn-hidden-dim", type=int, default=256)
    p.add_argument("--gnn-out-dim", type=int, default=512)
    p.add_argument("--num-gnn-layers", type=int, default=4)
    p.add_argument("--num-proj-layers", type=int, default=1)

    # Training parameters
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument(
        "--optim",
        type=str,
        choices=["lion", "adamw", "adafactor"],
        default="lion",
        help="Optimizer to use for SFT training.",
    )
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-scheduler-type", type=str, choices=["linear", "cosine"], default="linear")
    p.add_argument("--warmup-ratio", type=float, default=0)
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--per-device-eval-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42, help="Random seed for all RNGs and dataset shuffles.")

    # Checkpointing
    p.add_argument("--save-intermediate-models", action="store_true", help="Save intermediate models")
    p.add_argument("--save-interval-epochs", type=int, default=1, help="Save every N epochs")
    p.add_argument("--no-save", action="store_true", help="Do not save any model checkpoints")
    p.add_argument("--output-dir", type=str, default="outputs", help="Base output directory")

    # Evaluation after training
    p.add_argument("--do-eval", action="store_true", help="Run evaluation after training")
    p.add_argument("--num-eval-trials", type=int, default=1, help="Number of evaluation trials to run after training.")

    # Logging
    p.add_argument("--wandb", action="store_true", help="Use wandb logging")
    p.add_argument("--tags", type=str, nargs="*", default=[], help="Tags for wandb run.")

    # Parse and validate args
    args = p.parse_args()
    validate_args(args)

    # Args for GraphTokenLMConfig and SFTConfig
    glm_args = {
        "base_model": args.base_model,
        "gnn_type": args.gnn_type,
        "node_feat_dim": args.lpe_dim + (1 if args.use_degree_emb else 0),
        "pos_emb_dim": args.pos_emb_dim,
        "gnn_hidden_dim": args.gnn_hidden_dim,
        "gnn_out_dim": args.gnn_out_dim,
        "num_gnn_layers": args.num_gnn_layers,
        "num_graph_tokens": args.num_graph_tokens,
        "num_proj_layers": args.num_proj_layers,
        "num_max_nodes": args.num_max_nodes,
        "lpe_dim": args.lpe_dim,
        "use_degree_emb": args.use_degree_emb,
        "graph_pooling": args.graph_pooling,
    }
    sft_args = {
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "optim": args.optim,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_intermediate_models": args.save_intermediate_models,
        "save_interval_epochs": args.save_interval_epochs,
    }
    if args.optim in ["adamw"]:
        sft_args["weight_decay"] = args.weight_decay
    elif args.weight_decay > 0:
        print(f"[WARNING] --weight-decay is ignored when --optim {args.optim} is used.")

    # Remove overlapped args
    for attr in [*glm_args.keys(), *sft_args.keys(), "epochs", "lr"]:
        if hasattr(args, attr):
            delattr(args, attr)

    return glm_args, sft_args, args


def setup_run_context(
    dataset: str,
    subsets: list[str],
    use_wandb: bool,
    tags: list[str],
    output_dir: str,
    glm_args: dict,
    multitask: bool = False,
) -> tuple[str, str]:
    """Setup output directory and initialize wandb if needed.

    Parameters
    ----------
    dataset : str
        Dataset name for the current run.
    subsets : list[str]
        Subset name(s) for the current run.
    use_wandb : bool
        Whether to use wandb logging.
    tags : list[str]
        Tags for wandb run.
    output_dir : str
        Base output directory.
    glm_args : dict
        Arguments for GraphTokenLMConfig.

    Returns
    -------
    out_dir : str
        Path to the output directory for the current run.
    date_str : str
        Timestamp string for the current run.
    """
    date_str = datetime.now().strftime("%m%d-%H%M")
    if multitask:
        run_name = f"multitask_{date_str}"
        out_dir = os.path.join(output_dir, "multitask", date_str)
    else:
        subset = subsets[0]
        run_name = f"{subset}_{date_str}"
        out_dir = os.path.join(output_dir, subset, date_str)
    if use_wandb and is_main_process():
        if multitask:
            config = {"dataset": dataset, "subset": subsets, "glm_args": glm_args}
        else:
            config = {"dataset": dataset, "subset": subsets[0], "glm_args": glm_args}
        match dataset:
            case "MotifQA":
                wandb.init(project="MotifQA-GLM", name=run_name, config=config, tags=tags)
            case "GraphQA":
                wandb.init(project="GraphQA-GLM", name=run_name, config=config, tags=tags)

    return out_dir, date_str


def build_custom_dataset(
    subset: str,
    lpe_dim: int,
    use_degree_emb: bool = False,
    do_eval: bool = False,
) -> tuple[Dataset, Dataset, Dataset | None, int]:
    """Build custom dataset for training and evaluation.

    Parameters
    ----------
    subset : str
        Subset of the GraphQA dataset to use.
    lpe_dim : int
        Dimensionality of Laplacian positional embeddings.
    use_degree_emb : bool, default=False
        Whether to append node degree as an additional feature dimension.
    do_eval : bool, default=False
        Whether to prepare the test dataset for evaluation.

    Returns
    -------
    train_ds : Dataset
        Training dataset with `prompt`, `completion`, and `graph` columns.
    eval_ds : Dataset
        Evaluation dataset with `prompt`, `completion`, and `graph` columns.
    test_ds : Dataset or None
        Test dataset if `do_eval` is True, otherwise None.
    num_max_nodes : int
        Maximum number of nodes across all graphs in the dataset.
    """
    ds_dict = load_dataset(
        "json",
        data_dir=os.path.join("dataset", subset),
        data_files=(
            {"train": "train.jsonl", "validation": "eval.jsonl", "test": "test.jsonl"}
            if do_eval
            else {"train": "train.jsonl", "validation": "eval.jsonl"}
        ),
    )

    match subset:
        case "node_count":
            ans_label = "{} Thus, the answer is {}."

            def modify_dataset(example):
                if example["nodes"]:
                    node_str = (
                        "Nodes in the graph are "
                        + ", ".join(map(str, example["nodes"][:-1]))
                        + " and "
                        + str(example["nodes"][-1])
                        + "."
                    )
                else:
                    node_str = "There are no nodes in the graph."
                ans_digit = example["answer"].strip().split(".")[0]
                example["answer"] = ans_label.format(node_str, ans_digit)
                return add_graph_column(
                    example,
                    ds_name="GraphQA",
                    lpe_dim=lpe_dim,
                    use_degree_emb=use_degree_emb,
                )

        case "edge_count":
            ans_label = "{} Thus, the answer is {}."

            def modify_dataset(example):
                if example["edges"]:
                    edge_str = (
                        "Edges in the graph are "
                        + ", ".join(map(str, map(tuple, example["edges"][:-1])))
                        + " and "
                        + str(tuple(example["edges"][-1]))
                        + "."
                    )
                else:
                    edge_str = "There are no edges in the graph."
                ans_digit = example["answer"].strip().split(".")[0]
                example["answer"] = ans_label.format(edge_str, ans_digit)
                return add_graph_column(
                    example,
                    ds_name="GraphQA",
                    lpe_dim=lpe_dim,
                    use_degree_emb=use_degree_emb,
                )

        case "triangle_counting":
            ans_label = "{} Thus, the answer is {}."

            def modify_dataset(example):
                if example["triangles"]:
                    tri_str = (
                        "Triangles in the graph are "
                        + ", ".join(map(str, map(tuple, example["triangles"][:-1])))
                        + " and "
                        + str(tuple(example["triangles"][-1]))
                        + "."
                    )
                else:
                    tri_str = "There are no triangles in the graph."
                ans_digit = example["answer"].strip().split(".")[0]
                example["answer"] = ans_label.format(tri_str, ans_digit)
                return add_graph_column(
                    example,
                    ds_name="GraphQA",
                    lpe_dim=lpe_dim,
                    use_degree_emb=use_degree_emb,
                )

        case _:
            raise NotImplementedError(f"Custom dataset for {subset} is not implemented.")

    ds_dict = ds_dict.map(modify_dataset, remove_columns=ds_dict["train"].column_names, desc="Preprocessing")
    num_max_nodes = 20
    return ds_dict["train"], ds_dict["validation"], ds_dict["test"] if do_eval else None, num_max_nodes


def _build_graphqa_dataset(
    subsets: list[str],
    lpe_dim: int,
    use_degree_emb: bool = False,
    do_eval: bool = False,
    load_from_cache_file: bool = True,
    seed: int = 42,
) -> tuple[Dataset, Dataset, dict[str, Dataset] | None, int]:
    """Build GraphQA datasets for training and evaluation (multi-subset aware)."""

    def modify_dataset(example):
        return add_graph_column(
            example,
            ds_name="GraphQA",
            lpe_dim=lpe_dim,
            use_degree_emb=use_degree_emb,
        )

    cols = ["algorithm", "answer", "nedges", "nnodes", "question", "task_description", "text_encoding"]
    train_parts = []
    eval_parts = []
    test_parts = []
    for subset in subsets:
        train_parts.append(load_dataset("baharef/GraphQA", subset, split="zero_shot_train"))
        eval_parts.append(load_dataset("baharef/GraphQA", subset, split="zero_shot_validation"))
        if do_eval:
            test_parts.append(load_dataset("baharef/GraphQA", subset, split="zero_shot_test"))
    train_raw = concatenate_datasets(train_parts).shuffle(seed=seed)
    eval_raw = concatenate_datasets(eval_parts).shuffle(seed=seed)
    train_ds = train_raw.map(
        modify_dataset,
        remove_columns=cols,
        load_from_cache_file=load_from_cache_file,
        desc="Preprocessing train (multitask)",
    )
    eval_ds = eval_raw.map(
        modify_dataset,
        remove_columns=cols,
        load_from_cache_file=load_from_cache_file,
        desc="Preprocessing eval (multitask)",
    )
    test_ds_map = (
        {
            subset: test_raw.map(
                modify_dataset,
                remove_columns=cols,
                load_from_cache_file=load_from_cache_file,
                desc=f"Preprocessing test ({subset})",
            )
            for subset, test_raw in zip(subsets, test_parts)
        }
        if do_eval
        else None
    )
    num_max_nodes = 20
    return train_ds, eval_ds, test_ds_map, num_max_nodes


def _build_motifqa_dataset(
    subsets: list[str],
    lpe_dim: int,
    use_degree_emb: bool = False,
    do_eval: bool = False,
    load_from_cache_file: bool = True,
    seed: int = 42,
) -> tuple[Dataset, Dataset, dict[str, Dataset] | None, int]:
    """Build MotifQA datasets for training and evaluation (multi-subset aware)."""

    def modify_dataset(example):
        return add_graph_column(
            example,
            ds_name="MotifQA",
            lpe_dim=lpe_dim,
            use_degree_emb=use_degree_emb,
        )

    cols = ["response", "nodes", "edges", "nnodes", "nedges"]
    train_parts = []
    eval_parts = []
    test_parts = []
    num_max_nodes = 0
    for subset in subsets:
        train_raw = load_dataset("naos-ku/motif-qa", subset, split="train")
        eval_raw = load_dataset("naos-ku/motif-qa", subset, split="validation")
        train_parts.append(train_raw)
        eval_parts.append(eval_raw)
        num_max_nodes = max(num_max_nodes, max(train_raw["nnodes"]), max(eval_raw["nnodes"]))
        if do_eval:
            test_raw = load_dataset("naos-ku/motif-qa", subset, split="test")
            test_parts.append(test_raw)
            num_max_nodes = max(num_max_nodes, max(test_raw["nnodes"]))
    train_raw = concatenate_datasets(train_parts).shuffle(seed=seed)
    eval_raw = concatenate_datasets(eval_parts).shuffle(seed=seed)
    train_ds = train_raw.map(
        modify_dataset,
        remove_columns=cols,
        load_from_cache_file=load_from_cache_file,
        desc="Preprocessing train (multitask)",
    )
    eval_ds = eval_raw.map(
        modify_dataset,
        remove_columns=cols,
        load_from_cache_file=load_from_cache_file,
        desc="Preprocessing eval (multitask)",
    )
    test_ds_map = (
        {
            subset: test_raw.map(
                modify_dataset,
                remove_columns=cols,
                load_from_cache_file=load_from_cache_file,
                desc=f"Preprocessing test ({subset})",
            )
            for subset, test_raw in zip(subsets, test_parts)
        }
        if do_eval
        else None
    )
    return train_ds, eval_ds, test_ds_map, num_max_nodes


def build_dataset(
    dataset: str,
    subsets: list[str],
    lpe_dim: int,
    use_degree_emb: bool = False,
    do_eval: bool = False,
    load_from_cache_file: bool = True,
    seed: int = 42,
) -> tuple[Dataset, Dataset, dict[str, Dataset] | None, int]:
    """Build (possibly single-subset) multitask datasets for training and evaluation.

    Parameters
    ----------
    dataset : str
        Dataset name, either ``"GraphQA"`` or ``"MotifQA"``.
    subsets : list[str]
        Subset name(s) to include. A single subset is treated as a special case
        of multitask and is still processed via this function.
    lpe_dim : int
        Dimensionality of Laplacian positional embeddings.
    use_degree_emb : bool, default=False
        Whether to append node degree as an additional feature dimension.
    do_eval : bool, default=False
        Whether to prepare the test dataset(s) for evaluation.
    load_from_cache_file : bool, default=True
        Whether to load from cache files if available.
    seed : int, default=42
        Random seed used for dataset shuffling.

    Returns
    -------
    train_ds : Dataset
        Training dataset with `prompt`, `completion`, and `graph` columns.
    eval_ds : Dataset
        Validation dataset with `prompt`, `completion`, and `graph` columns.
    test_ds_map : dict[str, Dataset] | None
        Mapping from subset name to test dataset if ``do_eval`` is True,
        otherwise None.
    num_max_nodes : int
        Maximum number of nodes across all graphs in the dataset.
    """
    match dataset:
        case "GraphQA":
            return _build_graphqa_dataset(
                subsets,
                lpe_dim,
                use_degree_emb=use_degree_emb,
                do_eval=do_eval,
                load_from_cache_file=load_from_cache_file,
                seed=seed,
            )
        case "MotifQA":
            return _build_motifqa_dataset(
                subsets,
                lpe_dim,
                use_degree_emb=use_degree_emb,
                do_eval=do_eval,
                load_from_cache_file=load_from_cache_file,
                seed=seed,
            )
        case _:
            raise NotImplementedError(f"Dataset {dataset} is not supported.")


def train_glm(
    train_ds: datasets.Dataset,
    eval_ds: datasets.Dataset,
    out_dir: str,
    glm_args: dict,
    sft_args: dict,
    args: argparse.Namespace,
) -> GraphTokenLM:
    """
    Fine-tune a GraphToken language model on graph QA data using TRL's SFTTrainer.

    Parameters
    ----------
    train_ds : datasets.Dataset
        Training split in prompt-completion format. Each element must be a
        mapping with keys:
        - ``prompt`` (str): Instruction or question text.
        - ``completion`` (str): Target answer text.
        - ``graph`` (dict): Graph payload convertible to ``torch_geometric.data.Data``
          via :func:`src.collator.pyg_from_dict`. Expected fields are
          ``x`` (FloatTensor of shape ``(num_nodes, k)``), ``edge_index`` (LongTensor
          of shape ``(2, num_edges)`` with 0-based consecutive node indices), and
          optional ``num_nodes`` or ``edge_attr``. Extra keys are ignored.
    eval_ds : datasets.Dataset
        Validation split with the same schema as ``train_ds``. Can be ``None`` to
        skip evaluation steps.
    out_dir : str
        Directory where checkpoints and trainer state will be written.
    glm_args : dict
        Keyword arguments forwarded to :class:`src.glm.GraphTokenLMConfig`.
    sft_args : dict
        Keyword arguments forwarded to :class:`trl.SFTConfig`. ``optim`` and
        save-related flags are consumed here before building the config.
    args : argparse.Namespace
        Parsed CLI arguments containing logging options (e.g., wandb flag).

    Returns
    -------
    model : GraphTokenLM
        The fine-tuned model instance (on the local process).
    """
    # Initialize model and tokenizer
    glm_cfg = GraphTokenLMConfig(**glm_args)
    model = GraphTokenLM(glm_cfg)
    tokenizer = AutoTokenizer.from_pretrained(glm_cfg.base_model, trust_remote_code=True)

    # Compute save interval steps
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    micro_batches_per_epoch = ceil(len(train_ds) / (sft_args["per_device_train_batch_size"] * world_size))
    steps_per_epoch = ceil(micro_batches_per_epoch / sft_args["gradient_accumulation_steps"])

    save_intermediate_models = sft_args.pop("save_intermediate_models")
    save_interval_epochs = sft_args.pop("save_interval_epochs")
    if save_intermediate_models and is_main_process():
        print(f"[INFO] Intermediate models will be saved every {save_interval_epochs} epochs.")

    # Prepare optimizer mapping
    optim_choice = sft_args.pop("optim")
    hf_optim_map = {
        "lion": "lion_32bit",
        "adamw": "adamw_torch",
        "adafactor": "adafactor",
    }

    # Setup SFTTrainer
    save_strategy = "no" if args.no_save else ("steps" if save_intermediate_models else "no")
    sft_config = SFTConfig(
        optim=hf_optim_map[optim_choice],
        completion_only_loss=True,
        bf16=True,
        output_dir=out_dir,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        save_strategy=save_strategy,
        save_steps=steps_per_epoch * save_interval_epochs,
        report_to="wandb" if args.wandb else "none",
        remove_unused_columns=False,
        ddp_backend="nccl",  # DDP
        ddp_find_unused_parameters=False,  # since all params are used in each forward pass
        gradient_checkpointing=False,  # GraphTokenLM currently lacks gradient checkpoint support
        **sft_args,
    )
    collator = GraphQACollator(
        tokenizer=tokenizer,
        max_length=512,
        num_graph_tokens=glm_cfg.num_graph_tokens,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,  # Dataset should be `prompt-completion` format
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    # Start training
    if is_main_process():
        print("***** Training *****")
    trainer.train()

    # Save final model and trainer state
    if is_main_process() and not args.no_save:
        if not save_intermediate_models:
            final_step = trainer.state.global_step
            final_ckpt_dir = os.path.join(out_dir, f"checkpoint-{final_step}")
            trainer.save_model(final_ckpt_dir)
            print(f"[INFO] Final model saved at {final_ckpt_dir}.")
        trainer.save_state()

    if is_main_process():
        print("***** Done *****")

    return model


def eval_ddp(
    model,
    subset: str,
    test_ds: Dataset,
    max_new_tokens: int,
    date_str: str,
    use_wandb: bool,
    wandb_key: str,
    num_eval_trials: int = 1,
):
    """Evaluate the model on the test dataset in a DDP-aware manner.

    Parameters
    ----------
    model : GraphTokenLM
        The trained model to evaluate.
    subset : str
        Subset name for logging and result saving.
    test_ds : Dataset
        Test dataset to evaluate on.
    max_new_tokens : int
        Maximum number of tokens to generate during evaluation.
    date_str : str
        Timestamp string for result file naming.
    use_wandb : bool
        Whether to log results to wandb.
    wandb_key : str
        Key name for wandb logging.
    num_eval_trials : int, default=1
        Number of evaluation trials to run. Results are aggregated over trials.

    Returns
    -------
    acc : float or None
        Accuracy on the test dataset if running on the main process, otherwise None.
    """
    # Prepare dataset shard for each rank
    if dist.is_initialized():
        _safe_barrier()
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_test_ds = test_ds.shard(num_shards=world_size, index=rank)
    else:
        world_size = 1
        rank = 0
        local_test_ds = test_ds

    # Evaluate on the shard assigned to this rank.
    if is_main_process():
        print(f"[INFO] max_new_tokens={max_new_tokens}")
    all_results = [] if is_main_process() else None
    for _ in range(num_eval_trials):
        local_results = eval_model(model, local_test_ds, batch_size=8, max_new_tokens=max_new_tokens)
        if dist.is_initialized():
            gathered_results = [None] * world_size
            dist.all_gather_object(gathered_results, local_results)
            if rank == 0:
                all_results.extend([item for sublist in gathered_results for item in sublist])
        else:
            all_results.extend(local_results)

    # Collect and log results on the main process
    if is_main_process():
        res_file = os.path.join("results", subset, f"{date_str}.json")
        acc = collect_result(all_results, res_file, subset)
        if use_wandb and wandb_key:
            wandb.log({wandb_key: acc})
        return acc
    return None


def main():
    glm_args, sft_args, args = build_args()
    if is_main_process():
        print(f"Subsets: {', '.join(args.subset)}")
    multitask = len(args.subset) > 1
    test_ds_map = None

    # Fix seed for reproducibility
    set_seed(args.seed)

    # Load dataset
    if args.use_custom_dataset:
        subset = args.subset[0]
        if is_main_process():
            print("[INFO] Building dataset with custom prompt.")
        train_ds, eval_ds, test_ds, num_max_nodes = build_custom_dataset(
            subset,
            glm_args["lpe_dim"],
            use_degree_emb=glm_args["use_degree_emb"],
            do_eval=args.do_eval,
        )
        test_ds_map = {subset: test_ds} if (args.do_eval and test_ds is not None) else None
    else:
        train_ds, eval_ds, test_ds_map, num_max_nodes = build_dataset(
            args.dataset,
            args.subset,
            glm_args["lpe_dim"],
            use_degree_emb=glm_args["use_degree_emb"],
            do_eval=args.do_eval,
            load_from_cache_file=False,
            seed=args.seed,
        )

    # Update node capacity of GLM if needed
    if num_max_nodes > glm_args["num_max_nodes"]:
        glm_args["num_max_nodes"] = num_max_nodes
        if is_main_process():
            print(f"[INFO] Updated glm_args['num_max_nodes'] as {num_max_nodes}.")

    # Initialize wandb, setup output directory and date
    out_dir, date_str = setup_run_context(
        args.dataset,
        args.subset,
        use_wandb=args.wandb,
        tags=args.tags,
        output_dir=args.output_dir,
        glm_args=glm_args,
        multitask=multitask,
    )

    # Training
    model = train_glm(train_ds, eval_ds, out_dir, glm_args, sft_args, args)

    # Evaluate the trained model (multiple trials if requested)
    if args.do_eval and test_ds_map is not None:
        if is_main_process():
            print("***** Evaluation *****")
        max_new_tokens_dict = EXT_MAX_NEW_TOKENS if args.use_custom_dataset else MAX_NEW_TOKENS
        subset_avg_accs = []
        for subset in args.subset:
            max_new_tokens = max_new_tokens_dict.get(subset, 32)
            acc = eval_ddp(
                model,
                subset,
                test_ds_map[subset],
                max_new_tokens,
                date_str,
                use_wandb=args.wandb,
                wandb_key="test_acc" if len(args.subset) == 1 else f"acc_{subset}",
                num_eval_trials=args.num_eval_trials,
            )
            if acc is not None:
                subset_avg_accs.append(acc)
        if args.wandb and len(subset_avg_accs) > 1:
            wandb.log({"test_acc": sum(subset_avg_accs) / len(subset_avg_accs)})

    # Remove output dir for no-save mode
    if args.no_save and is_main_process():
        shutil.rmtree(out_dir)
        print(f"[INFO] Removed output directory `{out_dir}` since --no-save is set.")
        parent_dir = os.path.dirname(out_dir.rstrip(os.sep))
        if parent_dir and os.path.isdir(parent_dir) and not os.listdir(parent_dir):
            os.rmdir(parent_dir)
            print(f"[INFO] Removed empty parent directory `{parent_dir}`.")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
