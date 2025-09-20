import argparse

from datasets import load_dataset

from src.glm import GraphTokenLM
from src.preprocess import add_graph_column


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--subset",
        type=str,
        choices=["node_count", "edge_count", "cycle_check", "triangle_counting", "maximum_flow"],
        default="edge_count",
    )
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--num_graph_tokens", type=int, default=4)
    return p.parse_args()


def eval_model(model_path: str):
    glm = GraphTokenLM.from_pretrained(model_path)
    print(glm)


if __name__ == "__main__":
    args = build_args()

    # 各プロセスで同じデータをロードしてOK（TrainerがSamplerをDDP用に設定）
    train_ds = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_train")
    eval_ds = load_dataset(
        "baharef/GraphQA",
        args.subset,
        split="zero_shot_validation" if args.subset != "maximum_flow" else "zero_shot_test",
    )

    train_ds = train_ds.map(add_graph_column, desc="add_graph_column(train)")
    eval_ds = eval_ds.map(add_graph_column, desc="add_graph_column(eval)")

    eval_model(args.model_path)
