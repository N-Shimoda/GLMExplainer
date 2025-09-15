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
    - dataset の各サンプルに 'graph'（dict）と 'task_description'（学習テキスト）がある前提。
    - tokenizer を与えるとテキストをここでトークナイズし、与えない場合は
      事前にトークナイズ済み（'input_ids', 'attention_mask', 'labels' 等）とみなしてスタックのみ行う。
    - 返り値に 'graph_batch'（torch_geometric.data.Batch）を追加する。
    """

    tokenizer: Optional[PreTrainedTokenizerBase] = None
    text_field: str = "task_description"
    max_length: int = 512
    pad_to_multiple_of: Optional[int] = None
    # 生成対象外トークンのマスキングを行わない（SFTTrainer の標準処理に任せる）
    # 必要なら format/テンプレートに応じてここで -100 マスク処理を追加してください。

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
        # SFTTrainer は labels を自動生成することが多いが、
        # ここでも用意しておくと安全（単純に next-token 予測の教師にする）
        labels = toks.input_ids.clone()
        # pad を損失から除外
        labels[labels == self.tokenizer.pad_token_id] = -100
        toks["labels"] = labels
        return toks

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ---- 1) グラフのバッチ化 ----
        pyg_list = []
        # print("Features")
        # pprint(features, compact=True, width=120)
        for f in features:
            if "graph" not in f:
                raise KeyError("Example is missing 'graph'.")
            pyg_list.append(pyg_from_dict(f["graph"]))
        graph_batch = PygBatch.from_data_list(pyg_list)

        # ---- 2) テキスト（tokenize or stack）----
        batch: Dict[str, Any] = {"graph_batch": graph_batch}

        if self.tokenizer is not None:
            # 2-1) ここでトークナイズ
            if self.text_field not in features[0]:
                raise KeyError(
                    f"'{self.text_field}' not found in dataset features. "
                    "Set SFTConfig(dataset_text_field=...) accordingly or adjust collator.text_field."
                )
            texts = [f[self.text_field] for f in features]
            toks = self._tokenize_texts(texts)
            batch.update(toks)
            # pad_token_id はモデルへ渡しておくと便利（前処理で参照される場合がある）
            batch["pad_token_id"] = self.tokenizer.pad_token_id
        else:
            # 2-2) 既に tokenized 済み（SFTTrainer の前処理に任せるケース）
            #      テンソルでなければ tensor 化してからスタック
            def _stack(name: str, dtype=None):
                vals = [f[name] for f in features if name in f]
                if not vals:
                    return
                if not isinstance(vals[0], torch.Tensor):
                    t = torch.tensor(vals, dtype=dtype)
                else:
                    t = (
                        torch.nn.utils.rnn.pad_sequence(vals, batch_first=True, padding_value=0)
                        if vals[0].dim() == 1
                        else torch.stack(vals)
                    )
                batch[name] = t

            _stack("input_ids", dtype=torch.long)
            _stack("attention_mask", dtype=torch.long)
            _stack("labels", dtype=torch.long)
            if "pad_token_id" in features[0]:
                batch["pad_token_id"] = features[0]["pad_token_id"]

        return batch


__all__ = ["GraphQACollator", "pyg_from_dict"]
