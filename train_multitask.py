import argparse
import os
from datetime import datetime

import torch.distributed as dist

import wandb
from datasets import concatenate_datasets, load_dataset
from eval import collect_result, eval_model
from src.preprocess import add_graph_column
from train import train_glm

# 複数サブセットをまとめて学習するための対象一覧
subsets = ["node_count", "edge_count", "cycle_check", "triangle_counting"]


def is_main_process() -> bool:
    # torchrun / accelerate で RANK=0 がメイン
    return int(os.environ.get("RANK", "0")) == 0


def build_args():
    p = argparse.ArgumentParser()

    # Model architecture
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--gnn_type", type=str, default="GCN", choices=["GCN", "GAT", "GIN", "GraphSAGE"])
    p.add_argument("--num_graph_tokens", type=int, default=4)
    p.add_argument("--node_feat_dim", type=int, default=8)
    p.add_argument("--node_pos_dim", type=int, default=8)
    p.add_argument("--gnn_hidden_dim", type=int, default=128)
    p.add_argument("--gnn_out_dim", type=int, default=128)
    p.add_argument("--num_gnn_layers", type=int, default=2)

    # Training parameters
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.01)

    # Logging
    p.add_argument("--wandb", action="store_true", help="Use wandb logging")
    p.add_argument("--wandb_project", type=str, default="GraphQA-GLM")
    p.add_argument("--do_eval", action="store_true", help="Run evaluation after training")

    return p.parse_args()


def build_dataset(do_eval: bool = False):
    """
    subsets で定義された 5 つのサブセットから train_raw / eval_raw (/ test_raw) を読み込み、
    それぞれ連結した上で 1 つの train_ds / eval_ds (/ test_ds) を返す。

    既存の引数 subset は互換性のために残しているが、この関数内では subsets の内容を使用する。
    maximum_flow には validation split がないため、eval には test split を使用する。
    """

    def modify_dataset(example):
        return add_graph_column(example, k=args.node_feat_dim)

    cols = ["algorithm", "answer", "nedges", "nnodes", "question", "task_description", "text_encoding"]

    # 各サブセットの split を読み込み
    train_parts = []
    eval_parts = []
    test_parts = []

    for s in subsets:
        # train
        train_parts.append(load_dataset("baharef/GraphQA", s, split="zero_shot_train"))

        # eval: maximum_flow は validation が存在しないため test を eval として使用
        eval_split = "zero_shot_validation" if s != "maximum_flow" else "zero_shot_test"
        eval_parts.append(load_dataset("baharef/GraphQA", s, split=eval_split))

        # test は常に test split
        if do_eval:
            test_parts.append(load_dataset("baharef/GraphQA", s, split="zero_shot_test"))

    # 連結
    train_raw = concatenate_datasets(train_parts).shuffle(seed=42)
    eval_raw = concatenate_datasets(eval_parts).shuffle(seed=42)

    # 前処理（グラフ列の追加など）
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
    args = build_args()

    date_str = datetime.now().strftime("%m%d-%H%M")
    run_name = f"multitask_{date_str}"
    output_dir = os.path.join("outputs", "multitask", date_str)

    if args.wandb and is_main_process():
        wandb.init(project=args.wandb_project, name=run_name)

    # Training
    train_ds, eval_ds, test_ds_list = build_dataset(do_eval=args.do_eval)
    model = train_glm(train_ds, eval_ds, output_dir, args)

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
