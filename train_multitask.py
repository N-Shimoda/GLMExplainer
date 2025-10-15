import os
from datetime import datetime

import torch.distributed as dist
from datasets import Dataset, concatenate_datasets, load_dataset

import wandb
from eval import collect_result, eval_model
from src.preprocess import add_graph_column
from train import build_args, is_main_process, train_glm

# List of subsets targeted for joint training
subsets = ["node_count", "edge_count", "cycle_check", "triangle_counting"]


def build_dataset(node_feat_dim: int, do_eval: bool = False) -> tuple[Dataset, Dataset, list[Dataset] | None]:
    """
    Load train_raw / eval_raw (/ test_raw) from the subsets defined in `subsets`,
    concatenate them, and return unified train_ds / eval_ds (/ test_ds).

    The legacy argument `subset` is kept for backward compatibility, but this function
    uses the contents of `subsets`. Since maximum_flow lacks a validation split, use
    the test split for eval.

    Parameters
    ----------
    node_feat_dim : int
        Dimension of node features to be added.
    do_eval : bool, optional
        Whether to load test datasets for evaluation, by default False.

    Returns
    -------
    train_ds : Dataset
        Combined training dataset.
    eval_ds : Dataset
        Combined evaluation dataset.
    test_ds_list : list[Dataset] | None
        List of test datasets for each subset, or None if do_eval is False.
    """

    def modify_dataset(example):
        return add_graph_column(example, k=node_feat_dim)

    cols = ["algorithm", "answer", "nedges", "nnodes", "question", "task_description", "text_encoding"]

    # Load each split for every subset
    train_parts = []
    eval_parts = []
    test_parts = []

    for s in subsets:
        # train
        train_parts.append(load_dataset("baharef/GraphQA", s, split="zero_shot_train"))

        # eval: maximum_flow has no validation split, so use the test split for eval
        eval_split = "zero_shot_validation" if s != "maximum_flow" else "zero_shot_test"
        eval_parts.append(load_dataset("baharef/GraphQA", s, split=eval_split))

        # test always uses the test split
        if do_eval:
            test_parts.append(load_dataset("baharef/GraphQA", s, split="zero_shot_test"))

    # Concatenate splits
    train_raw = concatenate_datasets(train_parts).shuffle(seed=42)
    eval_raw = concatenate_datasets(eval_parts).shuffle(seed=42)

    # Preprocess data (add graph columns, etc.)
    train_ds = train_raw.map(modify_dataset, remove_columns=cols, desc="Preprocessing train (combined)")
    eval_ds = eval_raw.map(modify_dataset, remove_columns=cols, desc="Preprocessing eval (combined)")
    test_ds_list = (
        [
            test_raw.map(modify_dataset, remove_columns=cols, desc="Preprocessing test for subsets")
            for test_raw in test_parts
        ]
        if do_eval
        else None
    )

    return train_ds, eval_ds, test_ds_list


if __name__ == "__main__":
    glm_args, sft_args, args = build_args(multitask=True)

    date_str = datetime.now().strftime("%m%d-%H%M")
    run_name = f"multitask_{date_str}"
    output_dir = os.path.join("outputs", "multitask", date_str)

    if args.wandb and is_main_process():
        wandb.init(project=args.wandb_project, name=run_name)

    # Training
    train_ds, eval_ds, test_ds_list = build_dataset(
        glm_args["node_feat_dim"],
        do_eval=args.do_eval,
    )
    model = train_glm(train_ds, eval_ds, output_dir, glm_args, sft_args, args)

    # Evaluation
    if is_main_process() and args.do_eval:
        print("***** Evaluation *****")
        acc_li = []
        for i, subset in enumerate(subsets):
            print(f"  - {subset}")
            results = eval_model(model, test_ds_list[i], batch_size=8, subset=subset)
            res_file = os.path.join("results", subset, f"{date_str}.json")
            acc = collect_result(results, res_file, subset)
            acc_li.append(acc)
        if args.wandb:
            wandb.log({"test_acc": acc_li})

    if dist.is_initialized():
        dist.destroy_process_group()
