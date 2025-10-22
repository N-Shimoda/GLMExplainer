import argparse

import torch
from datasets import load_dataset
from torch_geometric.data import Batch as PygBatch
from transformers import AutoTokenizer, GenerationConfig

from eval import create_pyg_batch
from src.ckpt import _resolve_ckpt_path
from src.glm import GraphTokenLM
from src.preprocess import add_graph_column


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True, help="Path to the model checkpoint")
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
    ds = ds.map(
        lambda x: add_graph_column(x, k=node_feat_dim),
        remove_columns=["algorithm", "answer", "nedges", "nnodes", "question", "task_description", "text_encoding"],
    )
    return ds


def load_model(model_path: str):
    ckpt_path, run_name = _resolve_ckpt_path(model_path)

    model = GraphTokenLM.from_pretrained(ckpt_path, load_llm_weights=False)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    print(f"Loaded model from {ckpt_path} (run name: {run_name})")

    tokenizer = AutoTokenizer.from_pretrained(model.config.llm_name, trust_remote_code=True)
    return model, tokenizer


class GLMWrapper(torch.nn.Module):
    def __init__(self, model: GraphTokenLM, tokenizer: AutoTokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        self.graph = None
        self.input_text = None
        self.output_text = None

    def forward(self, x, edge_index):
        if self.input_text is None:
            raise ValueError("Input text is not set. Please set it using 'set_input_text' method.")
        pass

    def set_inputs(self, input_text: str, graph: PygBatch, gen_cfg: GenerationConfig):
        if not isinstance(graph, PygBatch):
            raise ValueError("Input graph must be a torch_geometric.data.Batch")
        if not isinstance(input_text, str):
            raise ValueError("Input text must be a string.")
        if input_text.strip() == "":
            raise ValueError("Input text cannot be empty.")

        self.input_text = input_text
        self.graph = graph

        input_ids = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**input_ids, graph=graph, generation_config=gen_cfg)
        self.output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print("output_text: ", self.output_text)


def main():
    args = build_args()
    model, tokenizer = load_model(args.model_path)

    dataset = build_dataset(args.subset, args.split, node_feat_dim=model.config.node_feat_dim)
    sample = dataset[args.sample_idx]
    pyg_batch = create_pyg_batch([sample["graph"]], device=model.device)
    print("Dataset: ", dataset)
    # print("Sample:", sample)

    wrapper = GLMWrapper(model, tokenizer)
    gen_cfg = GenerationConfig(
        max_new_tokens=8,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    wrapper.set_inputs(sample["prompt"], pyg_batch, gen_cfg)

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
