import math
from typing import Literal, Optional

import torch
from torch_geometric.data import Batch as PygBatch
from torch_geometric.data import Data as PygData
from transformers import AutoTokenizer, GenerationConfig

from src.glm import GraphTokenLM

VALID_BASELINE_GRAPH_TYPES = ["complete", "empty"]


class GLMWrapper(torch.nn.Module):
    def __init__(
        self,
        model: GraphTokenLM,
        tokenizer: AutoTokenizer,
        per_device_gen_batch_size: int = 4,
    ):
        """
        A wrapper class for GraphTokenLM to be compatible with PyG explanation API.

        Parameters
        ----------
        model : GraphTokenLM
            The GraphTokenLM model to be wrapped.
        tokenizer : AutoTokenizer
            The tokenizer corresponding to the LLM used in the model.
        per_device_gen_batch_size : int, optional
            The batch size per device for output generation, by default 4.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.input_text = None
        self.generated_ids = None
        self.relevant_idx: Optional[list[int]] = None
        self._graph_template: Optional[PygBatch] = None
        self.per_device_gen_batch_size = per_device_gen_batch_size

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """Pseudo forward method for explainer compatibility."""
        if self.input_text is None:
            raise ValueError("Input text is not set. Please run `gen_output` first.")
        if self._graph_template is None:
            raise ValueError("Graph template is not set. Please run `gen_output` first.")
        if self.generated_ids is None:
            raise ValueError("No generated output available. Please run `set_generated_ids` first.")

        # Preprocess text input (prompt + generated tokens)
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

        # Prepare graph input
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        data = PygData(x=x, edge_index=edge_index)
        data.num_nodes = x.size(0)
        data.batch = batch
        graph = PygBatch.from_data_list([data]).to(self.model.device)

        # Labels for loss calculation
        # NOTE: Setting -100 for prompt and graph tokens to ignore them in loss computation
        X_len = prompt_ids.size(1)
        labels = inputs["input_ids"].clone()
        labels[:, :X_len] = -100
        prefix_labels = torch.full(
            (1, self.model.config.num_graph_tokens), -100, dtype=labels.dtype, device=labels.device
        )
        labels = torch.cat([prefix_labels, labels], dim=1)

        # Forward pass
        outputs = self.model(**inputs, graph=graph, labels=labels)

        # Compute representative value
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        shift_log_probs = log_probs[:, :-1, :]
        shift_token_ids = inputs["input_ids"][:, 1:]
        token_log_probs = shift_log_probs.gather(dim=-1, index=shift_token_ids.unsqueeze(-1)).squeeze(-1)

        gen_len = generated_ids.size(1)
        output_log_probs = token_log_probs[:, -gen_len:-1] if gen_len > 0 else token_log_probs[:, :0]

        if output_log_probs.numel() == 0:
            cumulative_log_likelihood = torch.zeros((), device=self.model.device)
        else:
            if self.relevant_idx is not None and len(self.relevant_idx) > 0:
                relevant_idx = [idx for idx in self.relevant_idx if idx < output_log_probs.size(1)]
                if not relevant_idx:
                    cumulative_log_likelihood = torch.zeros((), device=self.model.device)
                else:
                    idx_tensor = torch.tensor(relevant_idx, device=output_log_probs.device, dtype=torch.long)
                    cumulative_log_likelihood = output_log_probs.index_select(1, idx_tensor).sum()
            else:
                cumulative_log_likelihood = output_log_probs.sum()

        return cumulative_log_likelihood

    def gen_output(
        self, input_text: str, graph: PygBatch, gen_cfg: GenerationConfig, num_trials: int = 1
    ) -> list[str]:
        """Sets the input text and generates output text based on the graph.

        Parameters
        ----------
        input_text : str
            The input prompt text.
        graph : torch_geometric.data.Batch
            The graph data in PyG Batch format.
        gen_cfg : GenerationConfig
            Configuration for text generation.
        num_trials : int, optional
            Number of generation trials to perform, by default 1.

        Returns
        -------
        output_text : list[str]
            The generated output text for each trial.
        """
        if not isinstance(input_text, str):
            raise ValueError("Input text must be a string.")
        if input_text.strip() == "":
            raise ValueError("Input text cannot be empty.")
        if num_trials < 1:
            raise ValueError("Number of trials must be at least 1.")

        self._graph_template = graph.clone()
        self.input_text = input_text

        input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        prompt_length = input_ids["input_ids"].shape[-1]

        output_texts: list[str] = []
        data_list = graph.to_data_list()
        if len(data_list) != 1 and len(data_list) != num_trials:
            raise ValueError("Graph batch must contain a single graph or match the number of trials.")

        for offset in range(0, num_trials, self.per_device_gen_batch_size):
            batch_size = min(self.per_device_gen_batch_size, num_trials - offset)
            batched_inputs = {k: v.repeat(batch_size, 1) for k, v in input_ids.items()}
            if len(data_list) == num_trials:
                graph_batch = PygBatch.from_data_list(
                    [data.clone() for data in data_list[offset : offset + batch_size]]
                )
            else:
                graph_batch = PygBatch.from_data_list([data_list[0].clone() for _ in range(batch_size)])
            outputs = self.model.generate(**batched_inputs, graph=graph_batch, generation_config=gen_cfg)
            generated_ids_batch = outputs[:, prompt_length:]
            for row in generated_ids_batch:
                if row.numel() == 0:
                    output_texts.append("")
                else:
                    output_texts.append(self.tokenizer.decode(row, skip_special_tokens=True))

        return output_texts

    def set_generated_ids(self, output_text: str):
        """Set the generated token ids to GLMWrapper given the output text.
        `gen_output` must be called before this method to set the input text and graph template.

        Parameters
        ----------
        output_text : str
            The custom output text to set for explanation.
        """
        if not isinstance(output_text, str):
            raise ValueError("Output text must be a string.")
        if output_text.strip() == "":
            raise ValueError("Output text cannot be empty.")
        if self.input_text is None or self._graph_template is None:
            raise ValueError(
                "Input text and graph template must be set before setting output. Please call `gen_output` first."
            )
        else:
            output_ids = self.tokenizer(output_text, return_tensors="pt")["input_ids"].squeeze(0)
            self.generated_ids = output_ids.to(self.model.device)

    def set_relevant_ids(
        self,
        baseline_graph_type: Literal["complete", "empty"],
        llr_threshold: float = 0.0,
        verbose: bool = False,
    ) -> list[int]:
        """Set relevant token ids based on log-likelihood ratio (LLR) between original and baseline graphs.
        `gen_output` and `set_generated_ids` must be called before this method.

        Parameters
        ----------
        baseline_graph_type : Literal["complete", "empty"]
            The type of baseline graph to use for comparison.
        llr_threshold : float, optional
            The LLR threshold above which tokens are considered relevant, by default 0.0.
        verbose : bool, optional
            Whether to print the token probability comparison table, by default False.

        Returns
        -------
        relevant_ids : list[int]
            The list of relevant token ids based on the LLR threshold.
            This function also sets the same indices to `self.relevant_idx`.
        """
        if self.input_text is None:
            raise ValueError("Input text is not set. Please run `gen_output` first.")
        if self._graph_template is None:
            raise ValueError("Graph template is not set. Please run `gen_output` first.")
        if self.generated_ids is None:
            raise ValueError("No generated output available. Please run `set_generated_ids` first.")
        if baseline_graph_type not in VALID_BASELINE_GRAPH_TYPES:
            raise ValueError(
                f"Invalid baseline_graph_type: {baseline_graph_type}. " f"Must be one of {VALID_BASELINE_GRAPH_TYPES}."
            )
        # Compute original token probabilities
        org_token_probs = self.comp_token_probs(
            prompt=self.input_text,
            completion=self.tokenizer.decode(self.generated_ids, skip_special_tokens=True),
            graph=self._graph_template,
        )

        # Compute baseline token probabilities
        base_graph = self._graph_template.clone()
        num_nodes = base_graph.num_nodes
        match baseline_graph_type:
            case "complete":
                edges = torch.combinations(torch.arange(num_nodes, device=base_graph.edge_index.device), r=2).t()
                base_graph["edge_index"] = torch.cat([edges, edges.flip(0)], dim=1)
            case "empty":
                base_graph["edge_index"] = torch.empty((2, 0), dtype=torch.long, device=base_graph.edge_index.device)
        base_token_probs = self.comp_token_probs(
            prompt=self.input_text,
            completion=self.tokenizer.decode(self.generated_ids, skip_special_tokens=True),
            graph=base_graph,
        )

        # Compute LLR and determine relevant tokens
        self.relevant_idx = []
        print_rows = []
        eps = 1e-12
        for idx, ((org_id, org_token, org_prob), (base_id, base_token, base_prob)) in enumerate(
            zip(org_token_probs, base_token_probs)
        ):
            if org_id != base_id or org_token != base_token:
                raise ValueError("Token sequences do not match between original and baseline runs.")
            llr = math.log(org_prob + eps) - math.log(base_prob + eps)
            if llr > llr_threshold:
                self.relevant_idx.append(idx)
            if verbose:
                print_rows.append(
                    f"{org_id:8d} | {org_token:12s} | {org_prob:15.8f} | "
                    f"{base_prob:15.8f} | {llr:10.6f} | {'*' if llr > llr_threshold else '':>8}"
                )

        # Print token probability comparison table
        if verbose:
            print("\n[INFO] Token probabilities comparison:")
            print(
                f"{'Token ID':>8} | {'Token':>12} | {'Original Prob.':>15}"
                f" | {'Baseline Prob.':>15} | {'LLR':>10} | {'Relevant':>8}"
            )
            print("-" * 85)
            print("\n".join(print_rows))

        return self.relevant_idx

    def comp_token_probs(self, prompt: str, completion: str, graph: PygBatch):
        """Compute token probabilities for the completion tokens given the prompt and graph."""
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        if not completion_ids:
            raise ValueError("Completion encodes to zero tokens. Provide a non-empty completion.")

        input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=self.model.device)
        attention_mask = torch.ones_like(input_ids)
        graph = graph.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, graph=graph)

        log_probs = torch.log_softmax(outputs.logits, dim=-1)[0]
        base_pos = self.model.config.num_graph_tokens + len(prompt_ids) - 1
        if base_pos < 0:
            raise ValueError("Prompt is empty and graph tokens are disabled; cannot score completion tokens.")

        token_rows = []
        for idx, token_id in enumerate(completion_ids):
            pos = base_pos + idx
            if pos >= log_probs.size(0):
                raise ValueError("Token position exceeds model logits length.")
            log_prob = log_probs[pos, token_id].item()
            token_str = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            token_rows.append((token_id, token_str, math.exp(log_prob)))

        return token_rows
