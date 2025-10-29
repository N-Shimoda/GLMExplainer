import argparse
import os
from datetime import datetime
from math import ceil
from typing import Optional

import torch.distributed as dist
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed
from trl import SFTConfig, SFTTrainer

import wandb
from eval import collect_result, eval_model, get_max_new_tokens
from src.collator import GraphQACollator
from src.ds_stats import completion_length_report
from src.glm import GraphTokenLM, GraphTokenLMConfig
from src.preprocess import add_graph_column


def is_main_process() -> bool:
    # RANK = 0 is the main process
    return int(os.environ.get("RANK", "0")) == 0


def build_args(*, multitask: bool = False):
    p = argparse.ArgumentParser()

    # General settings
    if not multitask:
        p.add_argument(
            "--subset",
            type=str,
            choices=["node_count", "edge_count", "cycle_check", "triangle_counting", "house_check"],
            default="edge_count",
        )
    p.add_argument("--do-eval", action="store_true", help="Run evaluation after training")
    p.add_argument("--use-custom-dataset", action="store_true", help="Use custom dataset with extended answer labels.")

    # Model architecture
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--gnn-type", type=str, default="GCN", choices=["GCN", "GAT", "GIN", "GraphSAGE"])
    p.add_argument("--num-max-nodes", type=int, default=20)
    p.add_argument("--num-graph-tokens", type=int, default=4)
    p.add_argument("--node-feat-dim", type=int, default=8)
    p.add_argument("--pos-emb-dim", type=int, default=8)
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
    p.add_argument("--per-device-eval-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--save-intermediate-models", action="store_true", help="Save intermediate models")
    p.add_argument("--save-interval-epochs", type=int, default=1, help="Save every N epochs")

    # Logging
    p.add_argument("--wandb", action="store_true", help="Use wandb logging")

    args = p.parse_args()

    # Args for GraphTokenLMConfig and SFTConfig
    glm_args = {
        "base_model": args.base_model,
        "gnn_type": args.gnn_type,
        "node_feat_dim": args.node_feat_dim,
        "pos_emb_dim": args.pos_emb_dim,
        "gnn_hidden_dim": args.gnn_hidden_dim,
        "gnn_out_dim": args.gnn_out_dim,
        "num_gnn_layers": args.num_gnn_layers,
        "num_graph_tokens": args.num_graph_tokens,
        "num_proj_layers": args.num_proj_layers,
        "num_max_nodes": args.num_max_nodes,
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


def setup_run_context(subset: str, use_wandb: bool):
    """Setup output directory and initialize wandb if needed.

    Parameters
    ----------
    subset : str
        Subset name for the current run.
    use_wandb : bool
        Whether to use wandb logging.

    Returns
    -------
    output_dir : str
        Path to the output directory for the current run.
    date_str : str
        Timestamp string for the current run.
    """
    date_str = datetime.now().strftime("%m%d-%H%M")
    run_name = f"{subset}_{date_str}"
    output_dir = os.path.join("outputs", subset, date_str)
    if use_wandb and is_main_process():
        match subset:
            case "house_check":
                wandb.init(project="MotifQA-GLM", name=run_name)
            case _:
                wandb.init(project="GraphQA-GLM", name=run_name)

    return output_dir, date_str


def build_dataset(
    subset: str, node_feat_dim: int, do_eval: bool = False, load_from_cache_file: bool = True
) -> tuple[Dataset, Dataset, Dataset | None]:
    """Build dataset for training and evaluation.

    Parameters
    ----------
    subset : str
        Subset of the GraphQA dataset to use.
    node_feat_dim : int
        Dimensionality of node features (k in Laplacian PE).
    do_eval : bool, default=False
        Whether to prepare the test dataset for evaluation.
    load_from_cache_file : bool, default=True
        Whether to load from cache file if available.

    Returns
    -------
    train_ds : Dataset
        Training dataset with `prompt`, `completion`, and `graph` columns.
    eval_ds : Dataset
        Evaluation dataset with `prompt`, `completion`, and `graph` columns.
    test_ds : Dataset or None
        Test dataset if `do_eval` is True, otherwise None.
    """

    def modify_dataset(example):
        return add_graph_column(example, k=node_feat_dim)

    cols = ["algorithm", "answer", "nedges", "nnodes", "question", "task_description", "text_encoding"]

    splits = {"train": "zero_shot_train", "validation": "zero_shot_validation"}
    if do_eval:
        splits["test"] = "zero_shot_test"

    raw_ds = load_dataset("baharef/GraphQA", subset, split=splits)
    processed_ds = raw_ds.map(
        modify_dataset,
        remove_columns=cols,
        load_from_cache_file=load_from_cache_file,
        desc="Preprocessing dataset",
    )

    train_ds = processed_ds["train"]
    eval_ds = processed_ds["validation"]
    test_ds = processed_ds["test"] if do_eval else None

    # Sync processes if running with DDP
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    return train_ds, eval_ds, test_ds


def build_motif_dataset(
    node_feat_dim: int, do_eval: bool = False, load_from_cache_file: bool = True
) -> tuple[Dataset, Dataset, Optional[Dataset]]:
    """Build house motif dataset for training and evaluation.

    Parameters
    ----------
    node_feat_dim : int
        Dimensionality of node features (k in Laplacian PE).
    do_eval : bool, default=False
        Whether to prepare the test dataset for evaluation.
    load_from_cache_file : bool, default=True
        Whether to load from cache file if available.

    Returns
    -------
    train_ds : Dataset
        Training dataset with `prompt`, `completion`, and `graph` columns.
    eval_ds : Dataset
        Evaluation dataset with `prompt`, `completion`, and `graph` columns.
    test_ds : Dataset | None
        Test dataset with `prompt`, `completion`, and `graph` columns, or None if not needed.

    """

    def modify_dataset(example):
        return add_graph_column(example, k=node_feat_dim, ds_name="motif-qa")

    # Load and preprocess the dataset
    splits = {"train": "train", "validation": "validation"}
    if do_eval:
        splits["test"] = "test"
    raw_ds = load_dataset("naos-ku/motif-qa", split=splits)

    processed_ds = raw_ds.map(modify_dataset, load_from_cache_file=load_from_cache_file)

    train_ds = processed_ds["train"]
    eval_ds = processed_ds["validation"]
    test_ds = processed_ds["test"] if do_eval else None

    return train_ds, eval_ds, test_ds


def build_custom_dataset(
    subset: str, node_feat_dim: int, do_eval: bool = False
) -> tuple[Dataset, Dataset, Dataset | None]:
    """Build custom dataset for training and evaluation.

    Parameters
    ----------
    subset : str
        Subset of the GraphQA dataset to use.
    node_feat_dim : int
        Dimensionality of node features (k in Laplacian PE).
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
                return add_graph_column(example, k=node_feat_dim)

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
                return add_graph_column(example, k=node_feat_dim)

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
                return add_graph_column(example, k=node_feat_dim)

        case _:
            raise NotImplementedError(f"Custom dataset for {subset} is not implemented.")

    ds_dict = ds_dict.map(modify_dataset, remove_columns=ds_dict["train"].column_names, desc="Preprocessing")

    # Aggregate completion lengths and delegate JSON and figure generation to the helper function
    if is_main_process():
        completion_length_report(ds_dict, subset, main_process=is_main_process())

    return ds_dict["train"], ds_dict["validation"], ds_dict["test"] if do_eval else None


def train_glm(train_ds, eval_ds, output_dir, glm_args, sft_args, args):
    glm_cfg = GraphTokenLMConfig(**glm_args)
    model = GraphTokenLM(glm_cfg)
    tokenizer = AutoTokenizer.from_pretrained(glm_cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("[INFO] Explicitly setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    collator = GraphQACollator(
        tokenizer=tokenizer,
        max_length=512,
        num_graph_tokens=glm_cfg.num_graph_tokens,
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    micro_batches_per_epoch = ceil(len(train_ds) / (sft_args["per_device_train_batch_size"] * world_size))
    steps_per_epoch = ceil(micro_batches_per_epoch / sft_args["gradient_accumulation_steps"])

    save_intermediate_models = sft_args.pop("save_intermediate_models")
    save_interval_epochs = sft_args.pop("save_interval_epochs")
    optim_choice = sft_args.pop("optim")

    if save_intermediate_models and is_main_process():
        print(f"[INFO] Intermediate models will be saved every {save_interval_epochs} epochs.")

    # Map CLI choices to HF/TRL optimizer identifiers
    hf_optim_map = {
        "lion": "lion_32bit",
        "adamw": "adamw_torch",
        "adafactor": "adafactor",
    }

    sft_config = SFTConfig(
        optim=hf_optim_map[optim_choice],
        completion_only_loss=True,
        bf16=True,
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        save_strategy="steps" if save_intermediate_models else "no",
        save_steps=steps_per_epoch * save_interval_epochs,
        report_to="wandb" if args.wandb else "none",
        remove_unused_columns=False,
        ddp_backend="nccl",  # DDP
        ddp_find_unused_parameters=False,  # since all params are used in each forward pass
        gradient_checkpointing=False,  # GraphTokenLM currently lacks gradient checkpoint support.
        **sft_args,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,  # Dataset should be `prompt-completion` format
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    if is_main_process():
        print("***** Training *****")
    trainer.train()

    # Save the final model
    final_step = trainer.state.global_step
    final_ckpt_dir = os.path.join(output_dir, f"checkpoint-{final_step}")
    if is_main_process():
        if not save_intermediate_models:
            trainer.save_model(final_ckpt_dir)
        trainer.save_state()
        print("***** Done *****")

    return model, final_ckpt_dir


def eval_ddp(model, subset: str, test_ds: Dataset, max_new_tokens: int, date_str: str, use_wandb: bool):
    if dist.is_initialized():
        dist.barrier()
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
    local_results = eval_model(model, local_test_ds, batch_size=8, max_new_tokens=max_new_tokens)

    if dist.is_initialized():
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, local_results)
        results = [item for sublist in gathered_results for item in sublist] if rank == 0 else None
    else:
        results = local_results

    if is_main_process():
        res_file = os.path.join("results", subset, f"{date_str}.json")
        acc = collect_result(results, res_file, subset)
        if use_wandb:
            wandb.log({"test_acc": acc})


def main():
    glm_args, sft_args, args = build_args()
    if is_main_process():
        print(f"Subset: {args.subset}")

    # Wandb initialization, output directory
    output_dir, date_str = setup_run_context(args.subset, args.wandb)

    # Fix seed for reproducibility
    set_seed(42)

    # Training
    if args.use_custom_dataset:
        if is_main_process():
            print("[INFO] Using custom dataset.")
        train_ds, eval_ds, test_ds = build_custom_dataset(
            args.subset,
            glm_args["node_feat_dim"],
            do_eval=args.do_eval,
        )
    elif args.subset in ["node_count", "edge_count", "cycle_check", "triangle_counting"]:
        train_ds, eval_ds, test_ds = build_dataset(
            args.subset,
            glm_args["node_feat_dim"],
            do_eval=args.do_eval,
            load_from_cache_file=False,
        )
    else:
        train_ds, eval_ds, test_ds = build_motif_dataset(
            glm_args["node_feat_dim"],
            do_eval=args.do_eval,
            load_from_cache_file=False,
        )
    # Save datasets locally as JSONL (only on the main process to avoid races)
    out_dir = os.path.join("ds_debug", args.subset)
    if is_main_process():
        os.makedirs(out_dir, exist_ok=True)
        train_ds.to_json(os.path.join(out_dir, "train.jsonl"), orient="records", lines=True)
        eval_ds.to_json(os.path.join(out_dir, "eval.jsonl"), orient="records", lines=True)
        if args.do_eval:
            test_ds.to_json(os.path.join(out_dir, "test.jsonl"), orient="records", lines=True)

    model, ckpt_path = train_glm(train_ds, eval_ds, output_dir, glm_args, sft_args, args)

    # Quick evaluation with 1 trial
    if args.do_eval and test_ds is not None:
        if is_main_process():
            print("***** Evaluation *****")
        max_new_tokens = get_max_new_tokens(args.subset, args.use_custom_dataset)
        eval_ddp(model, args.subset, test_ds, max_new_tokens, date_str, args.wandb)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
