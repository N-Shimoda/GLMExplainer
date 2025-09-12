import torch
import torch.nn as nn

# 例：PyTorch Geometric を使う GNN
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


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


class GraphTokenLM(PreTrainedModel):
    """
    GNN + (pool → project) で得た graph tokens を
    LLM の入力埋め込み（inputs_embeds）の先頭に連結して学習するモデル。
    """

    config_class = AutoConfig  # 形式上（必須ではない）

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        node_feat_dim: int = 128,
        gnn_hidden: int = 256,
        gnn_out: int = 512,
        num_gnn_layers: int = 2,
        num_graph_tokens: int = 4,
        freeze_llm: bool = True,
    ):
        llm = AutoModelForCausalLM.from_pretrained(llm_name)
        super().__init__(llm.config)
        self.llm = llm
        self.num_graph_tokens = num_graph_tokens

        # 1) GNN エンコーダ
        self.gnn = SimpleGCN(in_dim=node_feat_dim, hid_dim=gnn_hidden, out_dim=gnn_out, num_layers=num_gnn_layers)

        # 2) graph→token 射影
        self.tokenizer_head = GraphTokenizer(
            gnn_out_dim=gnn_out,
            llm_hidden_size=self.llm.config.hidden_size,
            num_graph_tokens=num_graph_tokens,
        )

        # 3) LLM を凍結（推奨）
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

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

        # ---- 連結（先頭に挿入）----
        new_inputs = torch.cat([graph_tokens, inputs_embeds], dim=1)  # [B, k+T, H]

        # attention_mask と labels を拡張
        if attention_mask is None:
            attention_mask = input_ids.ne(self.llm.config.pad_token_id).long()
        new_attention = torch.cat(
            [torch.ones((B, self.num_graph_tokens), dtype=attention_mask.dtype, device=self.device), attention_mask],
            dim=1,
        )
        if labels is not None:
            ignore = torch.full((B, self.num_graph_tokens), -100, dtype=labels.dtype, device=self.device)
            new_labels = torch.cat([ignore, labels], dim=1)
        else:
            new_labels = None

        return new_inputs, new_attention, new_labels

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
        ), "input_ids か inputs_embeds のいずれかが必要です"
        assert graph is not None, "graph（GNN入力）が必要です"

        inputs_embeds, attention_mask, labels = self._concat_graph_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
            graph=graph,
        )

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **{
                k: v
                for k, v in generate_kwargs.items()
                if k in {"use_cache", "past_key_values", "output_hidden_states", "output_attentions"}
            },
        )
        return out

    # 生成時のサポート（必要に応じて）
    def prepare_inputs_for_generation(
        self, input_ids=None, inputs_embeds=None, attention_mask=None, graph=None, **kwargs
    ):
        # 学習時と同様にグラフトークンを先頭へ連結して返す
        if inputs_embeds is None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        assert graph is not None, "generate 時も graph が必要です"
        inputs_embeds, attention_mask, _ = self._concat_graph_tokens(
            input_ids=None, attention_mask=attention_mask, labels=None, inputs_embeds=inputs_embeds, graph=graph
        )
        return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask, "graph": None}

    def generate(self, *args, **kwargs):
        # HF generate をそのまま使えるように委譲（prepare_inputs_for_generation を利用）
        return self.llm.generate(*args, **kwargs)
