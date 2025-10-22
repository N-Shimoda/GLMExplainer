import argparse

import torch
from datasets import load_dataset
from torch_geometric.explain import Explainer, GNNExplainer
from transformers import AutoTokenizer

from src.ckpt import _resolve_ckpt_path
from src.glm import GraphTokenLM
from src.preprocess import add_graph_column


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    p.add_argument(
        "--subset",
        type=str,
        choices=["node_count", "edge_count", "cycle_check", "triangle_counting"],
        default="node_count",
        help="Dataset subset to use",
    )
    p.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        default="test",
        help="Dataset split to use",
    )
    p.add_argument("--sample-idx", type=int, default=0, help="Sample index to explain")
    return p.parse_args()


def build_dataset(subset: str, split: str, node_feat_dim: int):
    ds = load_dataset("baharef/GraphQA", subset, split=f"zero_shot_{split}")
    ds = ds.map(lambda x: add_graph_column(x, k=node_feat_dim))
    return ds


def load_model(model_path: str):
    ckpt_path, run_name = _resolve_ckpt_path(model_path)
    model = GraphTokenLM.from_pretrained(ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(model.config.llm_name)
    print(f"Loaded model from {ckpt_path} (run name: {run_name})")
    return model, tokenizer


class GLMWrapper(torch.nn.Module):
    def __init__(self, model: GraphTokenLM):
        super().__init__()
        self.model = model
        self.input_text = None

    def forward(self, x, edge_index):
        pass

    def set_input_text(self, input_text: str):
        if not isinstance(input_text, str):
            raise ValueError("Input text must be a string.")
        if input_text.strip() == "":
            raise ValueError("Input text cannot be empty.")
        self.input_text = input_text


def main():
    args = build_args()
    model, tokenizer = load_model(args.model_path)
    wrapper = GLMWrapper(model)

    dataset = build_dataset(args.subset, args.split, node_feat_dim=model.config.node_feat_dim)
    sample = dataset[args.sample_idx]
    print("Sample:", sample)

    # explainer = Explainer(
    #     model=wrapper,
    #     algorithm=GNNExplainer(epochs=200),
    #     explanation_type="model",
    #     node_mask_type="attributes",
    #     edge_mask_type="object",
    #     model_config=dict(
    #         mode="binary_classification",
    #         task_level="graph",
    #         return_type="raw",
    #     ),
    # )
    # explanation = explainer(x=sample["graph"]["x"], edge_index=sample["graph"]["edge_index"])
    # print(explanation)


if __name__ == "__main__":
    main()
