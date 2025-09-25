import warnings

import torch
import torch.nn as nn
from accelerate import init_empty_weights  # noqa
from torch_geometric.nn import GCNConv, global_mean_pool
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
        llm_name="Qwen/Qwen3-4B-Instruct-2507",
        node_feat_dim=128,
        gnn_hidden=256,
        gnn_out=512,
        num_gnn_layers=2,
        num_graph_tokens=4,
        freeze_llm=True,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.llm_name = llm_name
        self.node_feat_dim = node_feat_dim
        self.gnn_hidden = gnn_hidden
        self.gnn_out = gnn_out
        self.num_gnn_layers = num_gnn_layers
        self.num_graph_tokens = num_graph_tokens
        self.freeze_llm = freeze_llm

        # generate 互換のためにフィールドを用意（後でモデル側で上書き）
        self.vocab_size = kwargs.get("vocab_size", None)
        self.pad_token_id = kwargs.get("pad_token_id", None)
        self.bos_token_id = kwargs.get("bos_token_id", None)
        self.eos_token_id = kwargs.get("eos_token_id", None)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    # transformers の generate/GenerationConfig 互換
    def get_text_config(self, decoder: bool | None = None, **kwargs):
        return self


class SimpleGCN(nn.Module):
    """最小限の GCN。ノード埋め込みを出力。"""

    def __init__(self, in_dim, hid_dim, out_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hid_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.act(x)
                x = self.dropout(x)
        return x  # [num_nodes, out_dim]


class GraphTokenizer(nn.Module):
    """
    Graph → k 個のグラフトークン（LLM埋め込み次元）へ。
    - pool: global_mean_pool（ノード）＋任意で追加プール（例：学習可能トークン数 k を project で生成）
    - project: MLP/Linear で (batch, k, hidden) に射影
    """

    def __init__(self, gnn_out_dim, llm_hidden_size, num_graph_tokens=4):
        super().__init__()
        self.num_graph_tokens = num_graph_tokens
        self.project = nn.Sequential(
            nn.Linear(gnn_out_dim, llm_hidden_size * num_graph_tokens),
        )
        # （任意）トークン位置用の学習可能埋め込み
        self.graph_pos = nn.Embedding(num_graph_tokens, llm_hidden_size)

    def forward(self, node_repr, batch_index):
        """
        node_repr: [N_nodes_total, gnn_out_dim]
        batch_index: [N_nodes_total]  各ノードがどのグラフに属するか
        """
        # グローバルプール（各グラフごとに 1 ベクトル）
        pooled = global_mean_pool(node_repr, batch_index)  # [B, gnn_out_dim]
        B = pooled.size(0)
        # k 個のトークンへ線形展開
        tokens = self.project(pooled)  # [B, k*H]
        Hk = tokens.view(B, self.num_graph_tokens, -1)  # [B, k, hidden]
        # 位置埋め込みを付与
        pos = self.graph_pos.weight.unsqueeze(0).expand(B, -1, -1)  # [B, k, hidden]
        Hk = Hk + pos
        return Hk  # [B, k, hidden]


class GraphTokenLM(PreTrainedModel, GenerationMixin):
    """
    GNN + (pool → project) で得た graph tokens を
    LLM の入力埋め込み（inputs_embeds）の先頭に連結して学習するモデル。
    """

    # _tied_weights_keys = ["llm.lm_head.weight"]
    # _keys_to_ignore_on_load_missing = [r"^llm\.lm_head\.weight$"]

    config_class = GraphTokenLMConfig
    base_model_prefix = "llm"

    def __init__(self, config: GraphTokenLMConfig, load_llm_weights: bool = True):

        super().__init__(config)

        # (重要) 内部 LLM は config から from_config で「空構造」を作る
        # 後で GraphTokenLM.from_pretrained() が全体の state_dict をロードする
        if load_llm_weights:
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.llm_name, trust_remote_code=True, tie_word_embeddings=True
            )
        else:
            warnings.warn(
                "Initialized LLM weights from scratch. If this is unintended, set load_llm_weights=True.",
                UserWarning,
            )
            llm_cfg = AutoConfig.from_pretrained(config.llm_name)
            self.llm = AutoModelForCausalLM.from_config(llm_cfg)

        self.num_graph_tokens = config.num_graph_tokens

        # 1) GNN エンコーダ
        self.gnn = SimpleGCN(
            in_dim=config.node_feat_dim,
            hid_dim=config.gnn_hidden,
            out_dim=config.gnn_out,
            num_layers=config.num_gnn_layers,
        )

        # 2) graph→token 射影
        self.tokenizer_head = GraphTokenizer(
            gnn_out_dim=config.gnn_out,
            llm_hidden_size=self.llm.config.hidden_size,
            num_graph_tokens=config.num_graph_tokens,
        )

        # 3) LLM を凍結（必要なら）
        if config.freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
            self.llm.eval()

        # --- sync basic generation fields so GenerationMixin works cleanly ---
        for k in ["vocab_size", "pad_token_id", "bos_token_id", "eos_token_id"]:
            setattr(self.config, k, getattr(self.llm.config, k, None))

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
        """
        graph: dict-like（collator が作る想定）
            - x: [N_nodes, node_feat_dim]
            - edge_index: [2, N_edges]
            - batch: [N_nodes]  各ノードのグラフID
        """
        # LLM の埋め込みを取得
        if inputs_embeds is None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        B, T, H = inputs_embeds.size()

        # ---- Graph → tokens ----
        x = graph["x"]  # [N_nodes, node_feat_dim]
        edge_index = graph["edge_index"]  # [2, N_edges]
        batch = graph["batch"]  # [N_nodes]
        node_repr = self.gnn(x, edge_index)  # [N_nodes, gnn_out]
        graph_tokens = self.tokenizer_head(node_repr, batch)  # [B, k, H]

        # print("graph_tokens.shape:", graph_tokens.shape)
        # print("inputs_embeds.shape:", inputs_embeds.shape)
        # print("input_embeds:", inputs_embeds)

        # ---- 連結（先頭に GraphToken を挿入）----
        new_inputs = torch.cat([graph_tokens, inputs_embeds], dim=1)  # [B, k+T, H]

        # attention_mask を k 個の 1 で前置
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
        labels=None,
        inputs_embeds=None,
        graph=None,
        **generate_kwargs,
    ) -> CausalLMOutputWithPast:
        assert (input_ids is not None) or (
            inputs_embeds is not None
        ), "Either input_ids or inputs_embeds must be provided"

        if graph is not None:
            inputs_embeds, attention_mask, labels = self._concat_graph_tokens(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                inputs_embeds=inputs_embeds,
                graph=graph,
            )
        else:
            # 生成時など、すでに graph tokens を結合済みの inputs_embeds が渡るケース
            if inputs_embeds is None:
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            if attention_mask is None:
                if input_ids is None:
                    raise ValueError("When attention_mask is not provided, input_ids must also be provided")
                attention_mask = input_ids.ne(self.llm.config.pad_token_id).long()

        # Qwen3 の損失に影響しうるキーは除外（安全側）
        blocked = {
            "num_items_in_batch",
            "label_smoothing",  # TRL/transformers が付けることがある
            "labels_shifted",  # 同上
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
        print("input_embeds", inputs_embeds.shape)
        print(inputs_embeds)
        print("attention_mask", attention_mask.shape)
        print(attention_mask)
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
