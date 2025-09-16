# src/collator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import Tensor

# PyTorch Geometric
from torch_geometric.data import Batch as PygBatch
from torch_geometric.data import Data as PygData
from transformers import PreTrainedTokenizerBase


def _to_tensor(x, dtype=None) -> Tensor:
    """
    安全に torch.Tensor 化。dtype が渡されれば強制変換。
    """
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype) if dtype is not None else x
    t = torch.tensor(x)
    return t.to(dtype=dtype) if dtype is not None else t


def pyg_from_dict(g: Dict[str, Any]) -> PygData:
    """
    create_pyg_dict()（src/preprocess.py）で作成した辞書を PYG Data に変換する。
    期待キー:
      - x:         (N, F)   float
      - edge_index:(2, E)   long
      - num_nodes: int      （なくても x から推定）
    任意:
      - edge_attr: (E, Fe)  float
    """
    if g is None:
        raise ValueError("graph is None")

    x = _to_tensor(g.get("x"), dtype=torch.float)
    edge_index = _to_tensor(g.get("edge_index"), dtype=torch.long)

    if x.dim() != 2:
        raise ValueError(f"x must be 2D (N,F), got shape={tuple(x.shape)}")
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be shape (2,E), got shape={tuple(edge_index.shape)}")

    num_nodes = g.get("num_nodes", x.shape[0])
    data = PygData(x=x, edge_index=edge_index, num_nodes=int(num_nodes))

    if "edge_attr" in g and g["edge_attr"] is not None:
        data.edge_attr = _to_tensor(g["edge_attr"], dtype=torch.float)

    return data


@dataclass
class GraphQACollator:
    """
    GraphQA 用のコラトラ。
    - 各サンプルに 'graph'（dict）と 'task_description'（学習テキスト）がある前提。
    - tokenizer があればここでトークナイズ、無ければ既に tokenized と見做してテンソル化のみ。
    - 返り値には 'graph'（PyG Batch）を入れる（モデルの forward が graph を受ける想定）。
    - SFTTrainer の compute_loss と整合するよう、labels を「k 個の -100 を先頭に前置」して長さを合わせる。
    """

    tokenizer: Optional[PreTrainedTokenizerBase] = None
    text_field: str = "task_description"
    max_length: int = 512
    pad_to_multiple_of: Optional[int] = None
    num_graph_tokens: int = 4  # ← モデルの k と一致させること

    # ---- 内部ユーティリティ ----
    def _tokenize_texts(self, texts: Sequence[str]) -> Dict[str, Tensor]:
        assert self.tokenizer is not None, "tokenizer is required to tokenize texts"
        toks = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        # labels を生成（pad は -100）
        labels = toks.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        toks["labels"] = labels
        return toks

    def _prepend_ignore_to_labels(self, labels: Tensor) -> Tensor:
        """
        labels の先頭に num_graph_tokens 個の -100 を前置して長さを (k+T) にする。
        既に k 分拡張済みなら何もしない（後方互換）。
        """
        if self.num_graph_tokens <= 0:
            return labels
        B, T = labels.shape
        # 既に拡張済み（例：他の前処理が先にやっている）ならスキップ
        # 判定: 全サンプルの先頭 k が -100 かつ次元が少なくとも k+1 ある
        k = self.num_graph_tokens
        if T > k and torch.all(labels[:, :k] == -100):
            return labels
        ignore = torch.full((B, k), -100, dtype=labels.dtype)
        return torch.cat([ignore, labels], dim=1)

    # ---- メイン ----
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1) グラフのバッチ化（CPU のまま返す）
        pyg_list = []
        for f in features:
            if "graph" not in f:
                raise KeyError("Example is missing 'graph'. Ensure add_graph_column() added it.")
            pyg_list.append(pyg_from_dict(f["graph"]))
        graph_batch = PygBatch.from_data_list(pyg_list)

        batch: Dict[str, Any] = {"graph": graph_batch}

        # 2) テキスト（tokenize or stack）
        if self.tokenizer is not None:
            if self.text_field not in features[0]:
                raise KeyError(
                    f"'{self.text_field}' not found in dataset features. "
                    "Set SFTConfig(dataset_text_field=...) or adjust collator.text_field."
                )
            texts = [f[self.text_field] for f in features]
            toks = self._tokenize_texts(texts)

            # ★ ラベルの長さを (k+T) に拡張（先頭に -100×k を追加）
            toks["labels"] = self._prepend_ignore_to_labels(toks["labels"])

            batch.update(toks)
            batch["pad_token_id"] = self.tokenizer.pad_token_id  # int でOK（CPU）
        else:
            # 既に tokenized 済み（input_ids/attention_mask/labels が入っている想定）
            def _stack(name: str, dtype=None):
                vals = [f[name] for f in features if name in f]
                if not vals:
                    return None
                if not isinstance(vals[0], torch.Tensor):
                    t = torch.tensor(vals, dtype=dtype)
                else:
                    t = (
                        torch.nn.utils.rnn.pad_sequence(vals, batch_first=True, padding_value=0)
                        if vals[0].dim() == 1
                        else torch.stack(vals)
                    )
                batch[name] = t
                return t

            input_ids = _stack("input_ids", dtype=torch.long)
            # attention_mask = _stack("attention_mask", dtype=torch.long)
            labels = _stack("labels", dtype=torch.long)

            # labels が無い場合は input_ids から生成（pad は 0 を想定／必要に応じて調整）
            if labels is None and input_ids is not None:
                labels = input_ids.clone()
                labels[labels == 0] = -100
                batch["labels"] = labels

            # ★ ここでも先頭に -100×k を追加
            if "labels" in batch:
                batch["labels"] = self._prepend_ignore_to_labels(batch["labels"])

            if "pad_token_id" in features[0]:
                batch["pad_token_id"] = features[0]["pad_token_id"]

        return batch


__all__ = ["GraphQACollator", "pyg_from_dict"]
