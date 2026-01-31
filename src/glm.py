from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    GraphSAGE,
    TransformerConv,
    global_add_pool,
    global_mean_pool,
)
from torch_geometric.utils import to_dense_batch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class GraphTokenLMConfig(PretrainedConfig):
    model_type = "graph_token_lm"

    def __init__(
        self,
        base_model="Qwen/Qwen3-4B-Base",
        gnn_type: Literal["GCN", "GAT", "GIN", "GraphSAGE", "GraphTransformer"] = "GCN",
        node_feat_dim=8,
        lpe_dim: int | None = None,
        use_degree_emb: bool = False,
        pos_emb_dim=8,
        gnn_hidden_dim=256,
        gnn_out_dim=512,
        num_gnn_layers=2,
        num_proj_layers=1,
        num_graph_tokens=4,
        num_max_nodes=20,  # maximum number of nodes per batch
        graph_pooling: Literal["mean", "sum"] = "mean",
        freeze_llm=True,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.base_model = base_model
        self.llm_name = base_model  # backward compatibility
        self.gnn_type = gnn_type

        self.node_feat_dim = node_feat_dim
        self.lpe_dim = lpe_dim if lpe_dim is not None else node_feat_dim
        self.use_degree_emb = bool(use_degree_emb)
        self.pos_emb_dim = pos_emb_dim
        self.node_pos_emb_dim = pos_emb_dim  # backward compatibility
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_hidden = gnn_hidden_dim  # backward compatibility
        self.gnn_out_dim = gnn_out_dim
        self.gnn_out = gnn_out_dim  # backward compatibility
        self.num_gnn_layers = num_gnn_layers
        self.num_proj_layers = num_proj_layers
        self.num_graph_tokens = num_graph_tokens
        self.num_max_nodes = num_max_nodes
        self.graph_pooling = graph_pooling
        self.freeze_llm = freeze_llm

        # Keep generation-related fields for compatibility (updated later).
        self.vocab_size = kwargs.get("vocab_size", None)
        self.pad_token_id = kwargs.get("pad_token_id", None)
        self.bos_token_id = kwargs.get("bos_token_id", None)
        self.eos_token_id = kwargs.get("eos_token_id", None)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    # Maintain compatibility with transformers.GenerationConfig.
    def get_text_config(self, decoder: bool | None = None, **kwargs):
        return self


class GNNEncoder(nn.Module):
    """Encode graph node features with a configurable GNN stack.

    Parameters
    ----------
    in_dim : int
        Dimensionality of the input node features.
    hid_dim : int
        Hidden dimensionality used for intermediate layers.
    out_dim : int
        Dimensionality of the output node representations.
    max_nodes : int
        Maximum number of nodes per graph in a batch.
    num_layers : int, default=2
        Number of graph convolution layers.
    node_pos_emb_dim : int, default=8
        Dimensionality of the optional learned positional embeddings.
    dropout : float, default=0.1
        Dropout probability applied between hidden layers.
    gnn_type : {"GCN", "GAT", "GIN", "GraphSAGE", "GraphTransformer"}, default="GCN"
        Type of graph convolution layer to build.
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        max_nodes: int,
        num_layers: int = 2,
        node_pos_emb_dim: int = 8,
        dropout: float = 0.1,
        gnn_type: Literal["GCN", "GAT", "GIN", "GraphSAGE", "GraphTransformer"] = "GCN",
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.pos_emb = nn.Embedding(max_nodes, node_pos_emb_dim) if node_pos_emb_dim > 0 else None

        in_channels = in_dim + (node_pos_emb_dim if node_pos_emb_dim > 0 else 0)
        if in_channels <= 0:
            raise ValueError("GNNEncoder requires a positive input feature dimension.")

        hidden_dims = [hid_dim] * max(num_layers - 1, 0)
        dims = [in_channels, *hidden_dims, out_dim]
        self.convs = nn.ModuleList()
        for i in range(len(dims) - 1):
            match gnn_type:
                case "GCN":
                    self.convs.append(GCNConv(dims[i], dims[i + 1]))
                case "GAT":
                    self.convs.append(GATConv(dims[i], dims[i + 1]))
                case "GIN":
                    self.convs.append(GINConv(nn.Linear(dims[i], dims[i + 1])))
                case "GraphSAGE":
                    self.convs.append(GraphSAGE(dims[i], dims[i + 1], 1))
                case "GraphTransformer":
                    # Multi-head attention with concat disabled to keep the output dim aligned.
                    self.convs.append(TransformerConv(dims[i], dims[i + 1], heads=4, concat=False, dropout=dropout))
                case _:
                    raise ValueError(f"Unsupported gnn_type: {gnn_type}")
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        if self.pos_emb is not None:
            # Assign positional indices per graph starting from zero within the batch.
            _, mask = to_dense_batch(x, batch, max_num_nodes=self.max_nodes)
            pos_idx = torch.arange(self.max_nodes, device=x.device).unsqueeze(0).expand(mask.size(0), -1)
            pos_idx = pos_idx[mask]
            x = torch.cat([x, self.pos_emb(pos_idx)], dim=-1)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.act(x)
                x = self.dropout(x)
        return x  # [num_nodes, out_dim]


class DomainProjector(nn.Module):
    """Project graph-level representations into graph tokens.

    The projector first pools node embeddings into a graph representation and
    then maps it into ``k`` graph tokens that match the language model's hidden
    dimension.

    Parameters
    ----------
    gnn_out_dim : int
        Dimensionality of the encoder output to project from.
    llm_hidden_size : int
        Target dimensionality matching the language model embeddings.
    num_graph_tokens : int, default=4
        Number of graph tokens to produce.
    graph_pooling : {"mean", "sum"}, default="mean"
        Pooling strategy used to aggregate node embeddings.
    num_layers : int, default=1
        Number of linear/GELU projection layers.
    """

    def __init__(self, gnn_out_dim, llm_hidden_size, num_graph_tokens=4, num_layers=1, graph_pooling: str = "mean"):
        super().__init__()
        if num_layers < 1:
            raise ValueError("DomainProjector requires at least one projection layer.")
        if graph_pooling not in {"mean", "sum"}:
            raise ValueError(f"Unsupported graph_pooling: {graph_pooling}")

        self.num_graph_tokens = num_graph_tokens
        self.graph_pooling = graph_pooling
        layers = []
        in_dim = gnn_out_dim
        for layer_idx in range(num_layers):
            out_dim = llm_hidden_size * num_graph_tokens if layer_idx == num_layers - 1 else gnn_out_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_idx < num_layers - 1:
                layers.append(nn.GELU())
            in_dim = out_dim
        self.project = nn.Sequential(*layers)
        # Optional learned positional embeddings for graph tokens.
        self.graph_pos = nn.Embedding(num_graph_tokens, llm_hidden_size)

    def forward(self, node_repr, batch_index):
        """Aggregate node embeddings into graph tokens.

        Parameters
        ----------
        node_repr : torch.Tensor
            Node representations of shape ``(num_nodes_total, gnn_out_dim)``.
        batch_index : torch.Tensor
            Batch indices identifying the graph for each node. Shape
            ``(num_nodes_total,)``.

        Returns
        -------
        torch.Tensor
            Graph token tensor of shape ``(batch_size, num_graph_tokens, hidden)``.
        """
        # Global pooling yields a single vector per graph.
        match self.graph_pooling:
            case "sum":
                pooled = global_add_pool(node_repr, batch_index)  # [B, gnn_out_dim]
            case "mean":
                pooled = global_mean_pool(node_repr, batch_index)  # [B, gnn_out_dim]
        B = pooled.size(0)

        # Expand into k tokens via the projection stack.
        tokens = self.project(pooled)  # [B, k*H]
        Hk = tokens.view(B, self.num_graph_tokens, -1)  # [B, k, hidden]

        # Add learned positional embeddings.
        pos = self.graph_pos.weight.unsqueeze(0).expand(B, -1, -1)  # [B, k, hidden]
        Hk = Hk + pos
        return Hk  # [B, k, hidden]


class GraphTokenLM(PreTrainedModel, GenerationMixin):
    """Language model that prepends graph tokens to textual inputs.

    A graph neural network encodes node features, pools them, and projects the
    result into learned graph tokens that are concatenated with language model
    embeddings before decoding.

    Parameters
    ----------
    config : GraphTokenLMConfig
        Model configuration describing the graph encoder and base LLM.
    load_llm_weights : bool, default=True
        Whether to load pretrained weights for the base language model.
    """

    _tied_weights_keys = ["llm.lm_head.weight"]
    _keys_to_ignore_on_load_missing = [r"^llm\.lm_head\.weight$"]

    config_class = GraphTokenLMConfig
    base_model_prefix = "llm"

    def __init__(self, config: GraphTokenLMConfig, load_llm_weights: bool = True):

        super().__init__(config)

        # LLM
        if load_llm_weights:
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.base_model, trust_remote_code=True, tie_word_embeddings=True
            )
        else:
            llm_cfg = AutoConfig.from_pretrained(config.base_model, dtype=torch.float32)
            self.llm = AutoModelForCausalLM.from_config(llm_cfg)

        self.num_graph_tokens = config.num_graph_tokens

        # GNN + Domain Projector
        self.gnn = GNNEncoder(
            gnn_type=config.gnn_type,
            node_pos_emb_dim=config.pos_emb_dim,
            in_dim=config.node_feat_dim,
            hid_dim=config.gnn_hidden_dim,
            out_dim=config.gnn_out_dim,
            num_layers=config.num_gnn_layers,
            max_nodes=config.num_max_nodes,
        )
        self.tokenizer_head = DomainProjector(
            gnn_out_dim=config.gnn_out_dim,
            llm_hidden_size=self.llm.config.hidden_size,
            num_graph_tokens=config.num_graph_tokens,
            num_layers=config.num_proj_layers,
            graph_pooling=config.graph_pooling,
        )

        if config.freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
            self.llm.eval()

        # --- sync basic generation fields so GenerationMixin works cleanly ---
        mirror_keys = [
            "vocab_size",
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
        ]
        for k in mirror_keys:
            if hasattr(self.llm.config, k):
                setattr(self.config, k, getattr(self.llm.config, k))

        # make sure tying is done once at init (harmless if already tied)
        if getattr(self.config, "tie_word_embeddings", False):
            self.tie_weights()

    @property
    def device(self):
        return next(self.parameters()).device

    def _concat_graph_tokens(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        inputs_embeds=None,
        graph=None,
    ):
        """Prepend graph tokens to language model embeddings.

        Parameters
        ----------
        input_ids : torch.Tensor, optional
            Token IDs used to derive embeddings if ``inputs_embeds`` is not provided.
        attention_mask : torch.Tensor, optional
            Attention mask aligned with ``input_ids``.
        labels : torch.Tensor, optional
            Label tensor passed through unchanged.
        inputs_embeds : torch.Tensor, optional
            Precomputed language model embeddings.
        graph : Mapping[str, torch.Tensor], optional
            Graph structure containing ``x``, ``edge_index``, and ``batch`` as
            produced by the collator.

        Returns
        -------
        tuple of torch.Tensor
            Tuple ``(inputs_embeds, attention_mask, labels)`` with graph tokens
            concatenated at the front of the sequence.
        """
        # Obtain embeddings from the base language model if necessary.
        if inputs_embeds is None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        B, T, H = inputs_embeds.size()

        # ---- Graph to tokens ----
        graph_device = next(self.gnn.parameters()).device
        if hasattr(graph, "to"):
            graph = graph.to(graph_device)

        x = graph["x"]  # [N_nodes, node_feat_dim]
        edge_index = graph["edge_index"]  # [2, N_edges]
        batch = graph["batch"]  # [N_nodes]
        node_repr = self.gnn(x, edge_index, batch)  # [N_nodes, gnn_out]
        graph_tokens = self.tokenizer_head(node_repr, batch)  # [B, k, H]
        if inputs_embeds is not None:
            graph_tokens = graph_tokens.to(inputs_embeds.device)

        # ---- Concatenate by prepending graph tokens ----
        new_inputs = torch.cat([graph_tokens, inputs_embeds], dim=1)  # [B, k+T, H]

        # Prepend ones to the attention mask for the graph tokens.
        if attention_mask is None:
            attention_mask = input_ids.ne(self.llm.config.pad_token_id).long()
        new_attention = torch.cat(
            [torch.ones((B, self.num_graph_tokens), dtype=attention_mask.dtype, device=self.device), attention_mask],
            dim=1,
        )

        return new_inputs, new_attention, labels

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        graph=None,
        labels=None,
        **generate_kwargs,
    ) -> CausalLMOutputWithPast:
        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        if graph is not None:
            inputs_embeds, attention_mask, labels = self._concat_graph_tokens(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                inputs_embeds=inputs_embeds,
                graph=graph,
            )
        else:
            # For generation, inputs may already include concatenated graph tokens.
            if inputs_embeds is None:
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            if attention_mask is None:
                if input_ids is None:
                    raise ValueError("When attention_mask is not provided, input_ids must also be provided")
                attention_mask = input_ids.ne(self.llm.config.pad_token_id).long()

        # Exclude auxiliary keys that may interfere with Qwen3 loss computation.
        blocked = {
            "num_items_in_batch",
            "label_smoothing",  # Added by TRL/transformers in some setups.
            "labels_shifted",  # Same as above.
        }
        passdown = {
            k: v
            for k, v in generate_kwargs.items()
            if k not in blocked and k in {"use_cache", "output_attentions", "output_hidden_states", "past_key_values"}
        }

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **passdown,
        )
        return out

    def prepare_inputs_for_generation(
        self, input_ids=None, inputs_embeds=None, attention_mask=None, graph=None, **kwargs
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Both input_ids and inputs_embeds cannot be provided at the same time")

        if inputs_embeds is None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        if graph is not None:
            inputs_embeds, attention_mask, _ = self._concat_graph_tokens(
                input_ids=None,
                attention_mask=attention_mask,
                labels=None,
                inputs_embeds=inputs_embeds,
                graph=graph,
            )

        return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask, "graph": None}

    # delegate embeddings to inner LLM so HF can tie weights correctly
    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.llm.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llm.set_output_embeddings(new_embeddings)

    def tie_weights(self):
        # honor config.tie_word_embeddings and delegate
        if getattr(self.config, "tie_word_embeddings", False):
            # inner LLM handles actual tying (lm_head <-> embeddings)
            self.llm.tie_weights()
        # keep parent behavior (no-op for most models)
        return super().tie_weights()
