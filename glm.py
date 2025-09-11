import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoModelForCausalLM, AutoTokenizer


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = self.dropout(h)
        # graph-level pooling（論文§3.1: グラフレベルは mean/sum pooling を採用）:contentReference[oaicite:7]{index=7}
        hg = global_mean_pool(h, batch)
        return h, hg  # node-level, graph-level


class GraphTokenHead(nn.Module):
    def __init__(self, gnn_hidden_dim, lm_embed_dim, n_graph_tokens=8):
        super().__init__()
        self.n_graph_tokens = n_graph_tokens
        self.proj = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, n_graph_tokens * lm_embed_dim),
        )

    def forward(self, hg):
        B, D = hg.shape
        out = self.proj(hg)  # (B, n_tokens * d_model)
        out = out.view(B, self.n_graph_tokens, -1)  # (B, n_tokens, d_model)
        return out


class GraphTokenLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm_path = "Qwen/Qwen3-4B-Instruct-2507"
        self.llm = AutoModelForCausalLM.from_pretrained(self.llm_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path)
        self.gnn = GraphEncoder(in_dim=4)
        self.dp = GraphTokenHead(gnn_hidden_dim=128, lm_embed_dim=2560)

    def forward(self, input_ids, attention_mask, x, edge_index, batch):
        # Encode graph
        _, hg = self.gnn(x, edge_index, batch)
        # Decode graph
        graph_tokens = self.dp(hg)
        # Prepare inputs for LLM
        inputs = self.tokenizer(input_ids, attention_mask=attention_mask, return_tensors="pt")
        # Forward through LLM
        outputs = self.llm(**inputs, graph_tokens=graph_tokens)
        return outputs
