from typing import Literal, Optional

import torch
from torch_geometric.data import Batch as PygBatch
from torch_geometric.data import Data as PygData
from transformers import AutoTokenizer, GenerationConfig

from src.glm import GraphTokenLM

VALIDE_AGGR_METHODS = ["normal"]


class GLMWrapper(torch.nn.Module):
    def __init__(self, model: GraphTokenLM, tokenizer: AutoTokenizer, aggr_method: Literal["normal"] = "normal"):
        """
        A wrapper class for GraphTokenLM to be compatible with PyG explanation API.

        Parameters
        ----------
        model : GraphTokenLM
            The GraphTokenLM model to be wrapped.
        tokenizer : AutoTokenizer
            The tokenizer corresponding to the LLM used in the model.
        aggr_method : Literal["normal"], optional
            The aggregation method for computing representative value, by default "normal".
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.input_text = None
        self.generated_ids = None
        self._graph_template: Optional[PygBatch] = None
        self.aggr_method = aggr_method

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

        # Compute representative value
        match self.aggr_method:
            case "normal":
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
            case _:
                raise NotImplementedError(f"Aggregation method '{self.aggr_method}' is not implemented.")

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

    def set_output(self, output_text: str):
        """Overwrites the generated output text with a custom output.
        `set_input` must be called before this method.

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
                "Input text and graph template must be set before setting output. Please call `set_input` first."
            )
        else:
            output_ids = self.tokenizer(output_text, return_tensors="pt")["input_ids"].squeeze(0)
            self.generated_ids = output_ids.to(self.model.device)

    def set_aggregation_method(self, method: Literal["normal"]):
        """Sets the aggregation method for computing representative value.

        Parameters
        ----------
        method : Literal["normal"]
            The aggregation method to use.
        """
        if method not in VALIDE_AGGR_METHODS:
            raise ValueError(f"Invalid aggregation method '{method}'. Valid methods are: {VALIDE_AGGR_METHODS}")
        self.aggr_method = method
