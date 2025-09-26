import argparse
import os
from datetime import datetime

import torch.distributed as dist
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

import wandb
from datasets import concatenate_datasets, load_dataset
from eval import collect_result, eval_model
from src.collator import GraphQACollator
from src.glm import GraphTokenLM, GraphTokenLMConfig
from src.preprocess import add_graph_column

# 複数サブセットをまとめて学習するための対象一覧
subsets = ["node_count", "edge_count", "cycle_check", "triangle_counting", "maximum_flow"]


def is_main_process() -> bool:
    # torchrun / accelerate で RANK=0 がメイン
    return int(os.environ.get("RANK", "0")) == 0


def build_args():
    p = argparse.ArgumentParser()

    # Model architecture
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--num_graph_tokens", type=int, default=4)
    p.add_argument("--node_feat_dim", type=int, default=4)
    p.add_argument("--gnn_hidden_dim", type=int, default=64)
    p.add_argument("--gnn_out_dim", type=int, default=64)
    p.add_argument("--num_gnn_layers", type=int, default=2)

    # Training parameters
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.05)

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
    train_raw = concatenate_datasets(train_parts)
    eval_raw = concatenate_datasets(eval_parts)
    test_raw = concatenate_datasets(test_parts) if do_eval else None

    # 前処理（グラフ列の追加など）
    train_ds = train_raw.map(modify_dataset, remove_columns=cols, desc="Preprocessing train (combined)")
    eval_ds = eval_raw.map(modify_dataset, remove_columns=cols, desc="Preprocessing eval (combined)")
    test_ds = (
        test_raw.map(modify_dataset, remove_columns=cols, desc="Preprocessing test (combined)") if do_eval else None
    )

    # 保存
    DS_DIR = "datasets"
    os.makedirs(DS_DIR, exist_ok=True)
    train_ds.to_json(os.path.join(DS_DIR, "train_ds.jsonl"))
    eval_ds.to_json(os.path.join(DS_DIR, "eval_ds.jsonl"))
    if do_eval and test_ds is not None:
        test_ds.to_json(os.path.join(DS_DIR, "test_ds.jsonl"))

    return train_ds, eval_ds, test_ds


def train_glm(train_ds, eval_ds, output_dir, args):
    glm_cfg = GraphTokenLMConfig(
        llm_name=args.base_model,
        node_feat_dim=args.node_feat_dim,
        num_graph_tokens=args.num_graph_tokens,
        gnn_hidden=args.gnn_hidden_dim,
        gnn_out=args.gnn_out_dim,
        num_gnn_layers=args.num_gnn_layers,
    )
    model = GraphTokenLM(glm_cfg)

    tokenizer = AutoTokenizer.from_pretrained(glm_cfg.llm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = GraphQACollator(
        tokenizer=tokenizer,
        max_length=512,
        num_graph_tokens=args.num_graph_tokens,
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        gradient_accumulation_steps=4,
        bf16=True,
        optim="lion_32bit",
        report_to="wandb" if args.wandb else "none",
        completion_only_loss=True,
        remove_unused_columns=False,
        ddp_backend="nccl",  # DDP
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    if is_main_process():
        print("***** Training *****")
    trainer.train()
    if is_main_process():
        print("***** Done *****")

    return model


if __name__ == "__main__":
    args = build_args()

    date_str = datetime.now().strftime("%m%d-%H%M")
    run_name = f"combined_{date_str}"
    output_dir = os.path.join("outputs", "combined", date_str)

    if args.wandb and is_main_process():
        wandb.init(project=args.wandb_project, name=run_name)

    # Training
    train_ds, eval_ds, test_ds = build_dataset(do_eval=args.do_eval)
    model = train_glm(train_ds, eval_ds, output_dir, args)

    # Evaluation
    if is_main_process() and args.do_eval:
        print("***** Evaluation *****")
        acc_li = []
        for subset in subsets:
            print(f"  - {subset}")
            results = eval_model(model, test_ds, batch_size=8, subset=subset)
            res_file = os.path.join("results", subset, f"{date_str}.json")
            acc = collect_result(results, res_file, subset)
            acc_li.append(acc)
        if args.wandb:
            wandb.log({"test_acc": acc_li})

    if dist.is_initialized():
        dist.destroy_process_group()
