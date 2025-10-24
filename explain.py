import argparse
import os
from typing import Optional

import torch
from datasets import arrow_dataset, load_dataset
from torch_geometric.data import Batch as PygBatch
from torch_geometric.data import Data as PygData
from torch_geometric.explain import Explainer, Explanation, GNNExplainer
from tqdm import tqdm
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
    # p.add_argument("--sample-idx", type=check_non_negative_int, default=0, help="Sample index to explain")
    return p.parse_args()


def build_dataset(subset: str, split: str, node_feat_dim: int) -> arrow_dataset.Dataset:
    """Builds and returns the specified dataset subset and split."""
    ds = load_dataset("baharef/GraphQA", subset, split=f"zero_shot_{split}")
    ds = ds.map(
        lambda x: add_graph_column(x, k=node_feat_dim),
        remove_columns=["algorithm", "answer", "nedges", "nnodes", "task_description", "text_encoding"],
    )
    ds = ds.add_column("index", list(range(len(ds))))
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


def save_explanation(explanation: Explanation, out_dir: str, sample_idx: int):
    """Save explanation visualizations to files.

    Parameters
    ----------
    explanation : torch_geometric.explain.Explanation
        The explanation object containing the results to visualize.
    out_dir : str
        Directory to save the explanation files.
    sample_idx : int
        Index of the sample being explained (used for file naming).
    """
    graph_path = os.path.join(out_dir, f"graph_{sample_idx}.pdf")
    feat_path = os.path.join(out_dir, f"feature_{sample_idx}.pdf")
    explanation.visualize_graph(graph_path)
    explanation.visualize_feature_importance(feat_path)
    print(f"Saved explanation graphs to\n\t- {graph_path}\n\t- {feat_path}")


class GLMWrapper(torch.nn.Module):
    def __init__(self, model: GraphTokenLM, tokenizer: AutoTokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.input_text = None
        self.generated_ids = None
        self._graph_template: Optional[PygBatch] = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """Pseudo forward method for explainer compatibility."""
        if self.input_text is None:
            raise ValueError("Input text is not set. Please run `set_input` first.")
        if self.generated_ids is None:
            raise ValueError("No generated output available. Please run `set_input` first.")

        # Text input
        prompt_inputs = self.tokenizer(self.input_text, return_tensors="pt").to(self.model.device)
        prompt_ids = prompt_inputs["input_ids"]
        generated_ids = self.generated_ids.to(self.model.device).unsqueeze(0)
        inputs = {"input_ids": torch.cat([prompt_ids, generated_ids], dim=1)}
        if "attention_mask" in prompt_inputs:
            gen_attention = torch.ones(
                (generated_ids.size(0), generated_ids.size(1)),
                dtype=prompt_inputs["attention_mask"].dtype,
                device=self.model.device,
            )
            inputs["attention_mask"] = torch.cat([prompt_inputs["attention_mask"], gen_attention], dim=1)

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
        X_len = prompt_ids.size(1)
        labels = inputs["input_ids"].clone()
        labels[:, :X_len] = -100
        prefix_labels = torch.full(
            (1, self.model.config.num_graph_tokens), -100, dtype=labels.dtype, device=labels.device
        )
        labels = torch.cat([prefix_labels, labels], dim=1)

        # Forward pass
        outputs = self.model(**inputs, graph=graph, labels=labels)

        # Compute log probs
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        shift_log_probs = log_probs[:, :-1, :]
        shift_token_ids = inputs["input_ids"][:, 1:]
        token_log_probs = shift_log_probs.gather(dim=-1, index=shift_token_ids.unsqueeze(-1)).squeeze(-1)

        gen_len = generated_ids.size(1)
        output_log_probs = token_log_probs[:, -gen_len:] if gen_len > 0 else token_log_probs[:, :0]

        if output_log_probs.numel() == 0:
            cumulative_log_likelihood = torch.zeros((), device=self.model.device)
            # log_prob_values = []
            # out_token_probs = []
        else:
            cumulative_log_likelihood = output_log_probs.sum()
            # output_log_probs_flat = output_log_probs.squeeze(0)
            # log_prob_values = output_log_probs_flat.detach().cpu().tolist()
            # out_token_probs = output_log_probs_flat.exp().detach().cpu().tolist()

        # generated_token_ids = self.generated_ids.detach().cpu().tolist()
        # generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_token_ids)
        # print("Output tokens:", [t.replace("Ġ", " ") for t in generated_tokens])
        # print("Sum of log probabilities:", cumulative_log_likelihood.item())
        # for t, p, lp in zip(generated_tokens, out_token_probs, log_prob_values):
        #     print(f"{t:>16s}: {p:.12f} (log={lp:.12f})")

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
        self.generated_ids = outputs[:, prompt_length:][0]

        if self.generated_ids.numel() == 0:
            output_text = ""
        else:
            output_text = self.tokenizer.decode(self.generated_ids, skip_special_tokens=True)
        return output_text


def explain_sample(wrapper: GLMWrapper, sample, pyg_batch: PygBatch, gen_cfg: GenerationConfig, MAX_TRIALS=10):

    correct = False
    generated = []
    for _ in range(MAX_TRIALS):
        output_text = wrapper.set_input(sample["prompt"], pyg_batch, gen_cfg)
        ans_val = sample["completion"].split(".")[0].strip()
        if ans_val in output_text:
            correct = True
            break
        else:
            generated.append(output_text)
    if not correct:
        print(f"[WARN] Failed to generate the correct answer after {MAX_TRIALS} trials (correct answer: {ans_val}).")
        print("Generated outputs:", generated)
        return None, None

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="graph",
            return_type="raw",
        ),
    )
    explanation = explainer(x=pyg_batch.x, edge_index=pyg_batch.edge_index, batch=pyg_batch.batch)
    return explanation, output_text


def main():
    args = build_args()

    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path)
    model.eval()

    # Load dataset
    dataset = build_dataset(args.subset, args.split, node_feat_dim=model.config.node_feat_dim)
    print("Dataset: ", dataset)

    # Create wrapper and generation config
    wrapper = GLMWrapper(model, tokenizer)
    gen_cfg = GenerationConfig(
        max_new_tokens=10,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    OUT_DIR = os.path.join("explanations", args.subset)
    os.makedirs(OUT_DIR, exist_ok=True)
    targets = dataset.filter(lambda x: int(x["completion"].split(".")[0]) == 1)
    for sample in tqdm(targets):
        pyg_batch = create_pyg_batch(sample["graph"], device=model.device)
        explanation, output_text = explain_sample(wrapper, sample, pyg_batch, gen_cfg)

        if explanation is not None:
            print(f"Question: `{sample['question']}`")
            print(f"Generated answer: `{output_text}`")
            print(f"Correct answer: `{sample['completion']}`")
            print(f"Explanation: {explanation}")
            save_explanation(explanation, OUT_DIR, sample_idx=sample["index"])
        else:
            print("explanation was None.")


if __name__ == "__main__":
    main()
