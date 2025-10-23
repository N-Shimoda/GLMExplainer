import argparse

import torch
from datasets import load_dataset
from torch_geometric.data import Batch as PygBatch
from torch_geometric.data import Data as PygData
from torch_geometric.explain import Explainer, GNNExplainer
from transformers import AutoTokenizer, GenerationConfig

from eval import create_pyg_batch
from src.ckpt import _resolve_ckpt_path
from src.glm import GraphTokenLM
from src.preprocess import add_graph_column


def build_args():
    def check_non_negative_int(value: str) -> int:
        try:
            ivalue = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"`{value}` is not an integer.")
        if ivalue < 0:
            raise argparse.ArgumentTypeError("Value must be non-negative.")
        return ivalue

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
    p.add_argument("--sample-idx", type=check_non_negative_int, default=0, help="Sample index to explain")
    return p.parse_args()


def build_dataset(subset: str, split: str, node_feat_dim: int):
    ds = load_dataset("baharef/GraphQA", subset, split=f"zero_shot_{split}")
    ds = ds.map(
        lambda x: add_graph_column(x, k=node_feat_dim),
        remove_columns=["algorithm", "answer", "nedges", "nnodes", "question", "task_description", "text_encoding"],
    )
    return ds


def load_model(model_path: str) -> tuple[GraphTokenLM, AutoTokenizer]:
    """Loads the GraphTokenLM model and tokenizer from the specified checkpoint path.

    Parameters
    ----------
    model_path : str
        Path to the task directory, model directory or a specific checkpoint.

    Returns
    -------
    model : GraphTokenLM
        Loaded GraphTokenLM model.
    tokenizer : AutoTokenizer
        Corresponding tokenizer used with the model.
    """
    ckpt_path, run_name = _resolve_ckpt_path(model_path)

    model = GraphTokenLM.from_pretrained(ckpt_path, load_llm_weights=False)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loaded model from {ckpt_path} (run name: {run_name})")

    tokenizer = AutoTokenizer.from_pretrained(model.config.llm_name, trust_remote_code=True)
    return model, tokenizer


class GLMWrapper(torch.nn.Module):
    def __init__(self, model: GraphTokenLM, tokenizer: AutoTokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.input_text = None
        self.output_text = None
        self._graph_template: PygBatch | None = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None = None):
        """Pseudo forward method for explainer compatibility."""
        if self.input_text is None:
            raise ValueError("Input text is not set. Please set it using 'set_input_text' method.")

        # Text input
        concat_text = self.input_text + self.output_text
        inputs = self.tokenizer(concat_text, return_tensors="pt").to(self.model.device)

        # Graph input
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self._graph_template is not None:
            graph = self._graph_template.clone()
            graph = graph.to(self.model.device)
            graph.x = x.to(self.model.device)
            graph.edge_index = edge_index.to(self.model.device)
            graph.batch = batch.to(self.model.device)
            graph.num_nodes = x.size(0)
        else:
            data = PygData(x=x, edge_index=edge_index)
            data.num_nodes = x.size(0)
            data.batch = batch
            graph = PygBatch.from_data_list([data]).to(self.model.device)

        # Labels for loss calculation
        X_len = len(self.tokenizer(self.input_text, return_tensors="pt")["input_ids"][0])
        labels = inputs["input_ids"].clone()
        labels[:, :X_len] = -100
        prefix_labels = torch.full(
            (1, self.model.config.num_graph_tokens), -100, dtype=labels.dtype, device=labels.device
        )
        labels = torch.cat([prefix_labels, labels], dim=1)

        outputs = self.model(**inputs, graph=graph, labels=labels)

        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        shift_log_probs = log_probs[:, :-1, :]
        shift_token_ids = inputs["input_ids"][:, 1:]
        token_log_probs = shift_log_probs.gather(dim=-1, index=shift_token_ids.unsqueeze(-1)).squeeze(-1)

        positions = torch.arange(shift_token_ids.size(1), device=token_log_probs.device)
        mask = positions >= (X_len - 1)
        output_log_probs = token_log_probs[:, mask]

        if output_log_probs.numel() == 0:
            cumulative_log_likelihood = torch.zeros((), device=self.model.device)
            log_prob_values = []
            out_token_probs = []
        else:
            cumulative_log_likelihood = output_log_probs.sum()
            output_log_probs_flat = output_log_probs.squeeze(0)
            log_prob_values = output_log_probs_flat.detach().cpu().tolist()
            out_token_probs = output_log_probs_flat.exp().detach().cpu().tolist()

        print("Output token probabilities:", out_token_probs)

        out_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        print("Output tokens:", out_tokens)
        print("Sum of log probabilities:", cumulative_log_likelihood.item())
        for t, p, lp in zip(out_tokens[X_len:], out_token_probs, log_prob_values):
            print(f"{t:>15s}: {p:.12f} (log={lp:.12f})")

        return cumulative_log_likelihood

    def set_input(self, input_text: str, graph: PygBatch, gen_cfg: GenerationConfig):
        """Sets the input text and generates output text based on the graph.

        Parameters
        ----------
        input_text : str
            The input prompt text.
        graph : torch_geometric.data.Batch
            The graph data in PyG Batch format.
        gen_cfg : GenerationConfig
            Configuration for text generation.

        Returns
        -------
        output_text : str
            The generated output text.
        """
        if not isinstance(input_text, str):
            raise ValueError("Input text must be a string.")
        if input_text.strip() == "":
            raise ValueError("Input text cannot be empty.")

        self._graph_template = graph.clone()
        self.input_text = input_text

        input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**input_ids, graph=graph, generation_config=gen_cfg)
        prompt_length = input_ids["input_ids"].shape[-1]
        generated_ids = outputs[:, prompt_length:]
        if generated_ids.numel() == 0:
            self.output_text = ""
        else:
            self.output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return self.output_text


def main():
    args = build_args()
    model, tokenizer = load_model(args.model_path)
    model.eval()

    dataset = build_dataset(args.subset, args.split, node_feat_dim=model.config.node_feat_dim)
    print("Dataset: ", dataset)

    sample = dataset[args.sample_idx]
    pyg_batch = create_pyg_batch(sample["graph"], device=model.device)

    wrapper = GLMWrapper(model, tokenizer)
    gen_cfg = GenerationConfig(
        max_new_tokens=8,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    output_text = wrapper.set_input(sample["prompt"], pyg_batch, gen_cfg)
    print(f"Generated output: `{output_text}`")
    ans_val = sample["completion"].split(".")[0].strip()
    if ans_val not in output_text:
        print(f"[INFO] The generated output does not contain the correct answer ({ans_val}).")
        return

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="binary_classification",
            task_level="graph",
            return_type="raw",
        ),
    )
    explanation = explainer(x=pyg_batch.x, edge_index=pyg_batch.edge_index, batch=pyg_batch.batch)
    print(explanation)
    explanation.visualize_graph(f"fig/explanation_{args.sample_idx}.pdf")
    explanation.visualize_feature_importance(f"fig/feature_importance_{args.sample_idx}.pdf")
    print("Correct answer:", sample["completion"])


if __name__ == "__main__":
    main()
