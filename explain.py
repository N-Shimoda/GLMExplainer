import argparse
import os
from typing import Optional

import torch
from datasets import arrow_dataset, load_dataset
from torch_geometric.data import Batch as PygBatch
from torch_geometric.data import Data as PygData
from torch_geometric.explain import Explainer, GNNExplainer
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig
from transformers.trainer_utils import set_seed

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
        "--dataset",
        type=str,
        choices=["MotifQA", "GraphQA"],
        default="MotifQA",
        help="Dataset name (default: MotifQA)",
    )
    p.add_argument(
        "--subset",
        type=str,
        choices=["node_count", "edge_count", "cycle_check", "triangle_counting"],
        help="Dataset subset to use. Only applicable for GraphQA.",
    )
    p.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        default="test",
        help="Dataset split to use",
    )
    p.add_argument(
        "--target-value", type=check_non_negative_int, default=None, help="Targeted answer value to explain"
    )
    p.add_argument("--sample-idx", type=check_non_negative_int, default=None, help="Index of the sample to explain")
    p.add_argument("--num-trials", type=int, default=1, help="Number of trials for explaining each sample")

    args = p.parse_args()

    # Validate arguments
    if args.target_value is not None and args.sample_idx is not None:
        raise ValueError("Only one of `target_value` or `sample_idx` should be specified.")
    if args.dataset == "MotifQA" and args.subset is not None:
        raise ValueError("`subset` argument is only applicable for GraphQA dataset.")
    if args.dataset == "GraphQA" and args.subset is None:
        raise ValueError("`subset` argument must be specified for GraphQA dataset.")

    return args


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
        output_log_probs = token_log_probs[:, -gen_len:-1] if gen_len > 0 else token_log_probs[:, :0]

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


def build_dataset(subset: str, dataset: str, split: str, node_feat_dim: int) -> arrow_dataset.Dataset:
    """Builds and returns the specified dataset subset and split."""
    match dataset:
        case "GraphQA":
            ds_raw = load_dataset("baharef/GraphQA", subset, split=f"zero_shot_{split}")
            ds = ds_raw.map(
                lambda x: add_graph_column(x, k=node_feat_dim, ds_name="GraphQA"),
                remove_columns=["algorithm", "answer", "nedges", "nnodes", "task_description", "text_encoding"],
                load_from_cache_file=False,
            )
        case "MotifQA":
            ds_raw = load_dataset("naos-ku/motif-qa", "yes_no", split=split)
            ds = ds_raw.map(
                lambda x: add_graph_column(x, k=node_feat_dim, ds_name="motif-qa"),
                remove_columns=["response", "nedges", "nnodes"],
                load_from_cache_file=False,
            )
    return ds.add_column("index", list(range(len(ds))))


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


def explain_sample(
    wrapper: GLMWrapper, sample: dict[str, str], pyg_batch: PygBatch, gen_cfg: GenerationConfig, MAX_TRIALS=10
):
    """Generates output for the given sample and explains it using GNNExplainer.

    Parameters
    ----------
    wrapper : GLMWrapper
        The model wrapper for GraphTokenLM.
    sample : dict
        A single dataset sample containing 'question' and 'completion'.
    pyg_batch : torch_geometric.data.Batch
        The graph data in PyG Batch format.
    gen_cfg : GenerationConfig
        Configuration for text generation.
    MAX_TRIALS : int, optional
        Maximum number of trials to generate the correct answer, by default 10.

    Returns
    -------
    explanation : torch_geometric.explain.Explanation
        The explanation object containing the results.
    output_text : str
        The generated output text.
    """
    # Generate output and verify correctness
    generated = []
    for _ in range(MAX_TRIALS):
        output_text = wrapper.set_input(sample["prompt"], pyg_batch, gen_cfg)
        generated.append(output_text)

    # Compute accuracy
    ans_val = sample["completion"].split(".")[0].strip()
    acc = sum(ans_val in out_text for out_text in generated) / len(generated)
    print(f"Generated outputs: {generated} (acc={acc:.2f})")
    if acc < 1.0:
        print(f"[WARN] Failed to generate the correct answer after {MAX_TRIALS} trials (correct answer: {ans_val}).")
        return None, None

    # Generate explanation by GNNExplainer
    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=200),
        # algorithm=CaptumExplainer("IntegratedGradients"),
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
    set_seed(42)
    args = build_args()

    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path)
    model.eval()

    # Load dataset
    dataset = build_dataset(args.subset, args.dataset, args.split, node_feat_dim=model.config.node_feat_dim)

    # Filter dataset samples to explain
    match args.dataset:
        case "MotifQA":
            filtered_ds = dataset.select(range(5))
            OUT_DIR = os.path.join("explanations", "house_check")
        case "GraphQA":
            if args.target_value is not None:
                filtered_ds = dataset.filter(lambda x: int(x["completion"].split(".")[0]) == args.target_value)
                TARGET_VALUE = args.target_value
            elif args.sample_idx is not None:
                filtered_ds = dataset.filter(lambda x: x["index"] == args.sample_idx)
                TARGET_VALUE = int(filtered_ds[0]["completion"].split(".")[0])
            OUT_DIR = os.path.join("explanations", f"{args.subset}_{TARGET_VALUE}")

    print("Dataset: ", filtered_ds)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Create wrapper and generation config
    wrapper = GLMWrapper(model, tokenizer)
    gen_cfg = GenerationConfig(
        max_new_tokens=10,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    for i in range(args.num_trials):
        # Compute explanations for each sample
        for sample in tqdm(filtered_ds):
            pyg_batch = create_pyg_batch(sample["graph"], device=model.device)
            explanation, output_text = explain_sample(wrapper, sample, pyg_batch, gen_cfg)

            if explanation is not None:
                print(f"Question: `{sample['prompt']}`")
                print(f"Generated answer: `{output_text}`")
                print(f"Correct answer: `{sample['completion']}`")
                print(f"Explanation: {explanation}")
                # Save explanation graphs
                suffix = f"{sample['index']}_{i}" if args.num_trials > 1 else f"{sample['index']}"
                graph_path = os.path.join(OUT_DIR, f"graph_{suffix}.svg")
                explanation.visualize_graph(graph_path)
                explanation.visualize_feature_importance(os.path.join(OUT_DIR, f"node_feat_{suffix}.svg"))
                print(f"Saved explanation graphs to {graph_path}")


if __name__ == "__main__":
    main()
