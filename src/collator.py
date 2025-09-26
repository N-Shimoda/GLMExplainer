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
    """安全に torch.Tensor 化。dtype が渡されれば強制変換。"""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype) if dtype is not None else x
    t = torch.tensor(x)
    return t.to(dtype=dtype) if dtype is not None else t


def pyg_from_dict(g: Dict[str, Any]) -> PygData:
    """
    Convert a dictionary to a PyTorch Geometric Data object.

    Parameters
    ----------
    g : dict
        Dictionary containing graph data. Expected keys are:
            x : array-like or Tensor, shape (N, F)
                Node features.
            edge_index : array-like or Tensor, shape (2, E)
                Edge indices.
            num_nodes : int, optional
                Number of nodes. If not provided, inferred from x.
            edge_attr : array-like or Tensor, shape (E, Fe), optional
                Edge attributes.

    Returns
    -------
    torch_geometric.data.Data
        PyTorch Geometric Data object containing the graph.
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
    - 各サンプルに 'graph'（dict）と 'prompt'/'completion' テキストがある前提。
    - tokenizer があればここでトークナイズ、無ければ既に tokenized と見做してテンソル化のみ。
    - 返り値には 'graph'（PyG Batch）を入れる（モデルの forward が graph を受ける想定）。
    - SFTTrainer / Transformers の損失と整合させるため、labels と attention_mask を
      GraphToken の個数 k だけ先頭拡張して (k+T) に揃える。
    """

    tokenizer: Optional[PreTrainedTokenizerBase] = None
    prompt_field: str = "prompt"
    completion_field: str = "completion"
    max_length: int = 512
    pad_to_multiple_of: Optional[int] = None
    num_graph_tokens: int = 4  # ← モデルの k と一致させること
    add_eos_token: bool = True

    # ---- 内部ユーティリティ ----
    def _encode(self, text: str) -> List[int]:
        assert self.tokenizer is not None, "tokenizer is required to tokenize texts"
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _build_prompt_completion(self, prompts: Sequence[str], completions: Sequence[str]) -> Dict[str, Tensor]:
        assert self.tokenizer is not None, "tokenizer is required to tokenize texts"

        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        if pad_token_id is None:
            raise ValueError("Tokenizer must define either pad_token_id or eos_token_id")

        input_ids_per_sample: List[List[int]] = []
        prompt_lengths: List[int] = []

        for prompt, completion in zip(prompts, completions):
            prompt_ids = self._encode(prompt)
            completion_ids = self._encode(completion)

            if self.add_eos_token and self.tokenizer.eos_token_id is not None:
                if not completion_ids or completion_ids[-1] != self.tokenizer.eos_token_id:
                    completion_ids.append(self.tokenizer.eos_token_id)

            combined = prompt_ids + completion_ids
            prompt_len = len(prompt_ids)

            if self.max_length and self.max_length > 0 and len(combined) > self.max_length:
                overflow = len(combined) - self.max_length

                # プロンプト側から優先的に切り詰める
                if overflow >= prompt_len:
                    overflow -= prompt_len
                    prompt_ids = []
                    prompt_len = 0
                else:
                    prompt_ids = prompt_ids[overflow:]
                    prompt_len = len(prompt_ids)
                    overflow = 0

                if overflow > 0:
                    completion_ids = completion_ids[overflow:]

                combined = (prompt_ids + completion_ids)[: self.max_length]
                prompt_len = min(prompt_len, len(combined))

            input_ids_per_sample.append(combined)
            prompt_lengths.append(prompt_len)

        max_seq_len = max((len(ids) for ids in input_ids_per_sample), default=0)
        if self.pad_to_multiple_of and max_seq_len % self.pad_to_multiple_of != 0:
            max_seq_len = (
                (max_seq_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of

        padded_input_ids: List[List[int]] = []
        padded_attention: List[List[int]] = []
        padded_labels: List[List[int]] = []

        for ids, prompt_len in zip(input_ids_per_sample, prompt_lengths):
            pad_len = max_seq_len - len(ids)
            padded_ids = ids + [pad_token_id] * pad_len
            attention = [1] * len(ids) + [0] * pad_len

            label_ids = [-100] * prompt_len
            label_ids.extend(ids[prompt_len:])
            label_ids.extend([-100] * pad_len)

            padded_input_ids.append(padded_ids)
            padded_attention.append(attention)
            padded_labels.append(label_ids)

        result = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "pad_token_id": int(pad_token_id),
        }
        return result

    def _prepend_ignore_to_toks(
        self,
        *,
        labels: Tensor,
        attention_mask: Tensor,
        orig_len: int,
    ) -> Dict[str, Tensor]:
        """
        labels と attention_mask を GraphToken (k) ぶんだけ「先頭に」前置する。
        - labels: 先頭に -100 を k 個
        - attention_mask: 先頭に 1 を k 個
        すでに (k+T) 長に拡張済みなら二重前置はしない。
        """
        k = int(self.num_graph_tokens)
        if k <= 0:
            return {"labels": labels, "attention_mask": attention_mask}

        B = labels.size(0)
        # --- labels ---
        if labels.size(1) == orig_len + k:
            new_labels = labels  # すでに拡張済み
        elif labels.size(1) == orig_len:
            ignore = torch.full((B, k), -100, dtype=labels.dtype)
            new_labels = torch.cat([ignore, labels], dim=1)
        else:
            # 想定外の長さ（例：テンプレ変更）でも強行はせず明示的に失敗させる
            raise ValueError(f"[collator] labels length {labels.size(1)} not in {{T={orig_len}, T+k={orig_len + k}}}")

        # --- attention_mask ---
        if attention_mask.size(1) == orig_len + k:
            new_attn = attention_mask  # すでに拡張済み
        elif attention_mask.size(1) == orig_len:
            ones = torch.ones((B, k), dtype=attention_mask.dtype)
            new_attn = torch.cat([ones, attention_mask], dim=1)
        else:
            raise ValueError(
                "[collator] attention_mask length {}"
                " not in {{T={}, T+k={}}}".format(attention_mask.size(1), orig_len, orig_len + k)
            )

        return {"labels": new_labels, "attention_mask": new_attn}

    # ---- メイン ----
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1) グラフのバッチ化（CPU のまま返す）
        pyg_list = [pyg_from_dict(f["graph"]) for f in features]
        graph_batch = PygBatch.from_data_list(pyg_list)

        batch: Dict[str, Any] = {"graph": graph_batch}

        # 2) テキスト（tokenize or stack）
        if self.tokenizer is not None:
            if self.prompt_field not in features[0]:
                raise KeyError(f"'{self.prompt_field}' not found in dataset features")
            if self.completion_field not in features[0]:
                raise KeyError(f"'{self.completion_field}' not found in dataset features")

            prompts = [f[self.prompt_field] for f in features]
            completions = [f[self.completion_field] for f in features]
            toks = self._build_prompt_completion(prompts, completions)

            # ★ labels / attention_mask を (k+T) に拡張（先頭前置）
            T = toks["input_ids"].size(1)
            padded = self._prepend_ignore_to_toks(
                labels=toks["labels"],
                attention_mask=toks["attention_mask"],
                orig_len=T,
            )
            toks["labels"] = padded["labels"]
            toks["attention_mask"] = padded["attention_mask"]

            batch.update(toks)
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
            attention_mask = _stack("attention_mask", dtype=torch.long)
            labels = _stack("labels", dtype=torch.long)

            # labels が無い場合は input_ids から生成（pad=0 を -100 に）
            if labels is None and input_ids is not None:
                labels = input_ids.clone()
                labels[labels == 0] = -100
                batch["labels"] = labels

            # attention_mask が無い場合も input_ids から生成（pad=0 を 0, それ以外 1）
            if attention_mask is None and input_ids is not None:
                attention_mask = (input_ids != 0).long()
                batch["attention_mask"] = attention_mask

            # ★ ここでも (k+T) に拡張（先頭前置）
            if ("labels" in batch) and ("attention_mask" in batch):
                T = input_ids.size(1)
                padded = self._prepend_ignore_to_toks(
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                    orig_len=T,
                )
                batch["labels"] = padded["labels"]
                batch["attention_mask"] = padded["attention_mask"]

            if "pad_token_id" in features[0]:
                batch["pad_token_id"] = int(features[0]["pad_token_id"])

        return batch


__all__ = ["GraphQACollator", "pyg_from_dict"]
