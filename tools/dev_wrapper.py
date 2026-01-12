import argparse
import os
import sys

import torch
from transformers import AutoTokenizer, GenerationConfig

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from eval import create_pyg_batch  # noqa: E402
from src.ckpt import _resolve_ckpt_path  # noqa: E402
from src.constants import GRAPHQA_SUBSETS, MOTIFQA_SUBSETS  # noqa: E402
from src.explanation.preprocess import build_dataset, filter_dataset  # noqa: E402
from src.explanation.wrapper import GLMWrapper  # noqa: E402
from src.glm import GraphTokenLM  # noqa: E402


def _load_model(model_path: str, device: torch.device) -> tuple[GraphTokenLM, AutoTokenizer]:
    ckpt_path, run_name = _resolve_ckpt_path(model_path)
    model = GraphTokenLM.from_pretrained(ckpt_path, load_llm_weights=False).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model.config.llm_name, trust_remote_code=True)
    print(f"[INFO] Loaded model from {ckpt_path} (run name: {run_name})")
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate GLMWrapper.set_relevant_ids() behavior.")
    p.add_argument("--model-path", type=str, default="outputs/ba_shapes")
    p.add_argument("--dataset", type=str, default="MotifQA", choices=["MotifQA", "GraphQA"])
    p.add_argument("--subset", type=str, default="ba_shapes", choices=MOTIFQA_SUBSETS + GRAPHQA_SUBSETS)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--sample-idx", type=int, default=None)
    p.add_argument("--baseline-graph", type=str, default="complete", choices=["complete", "empty"])
    p.add_argument("--max-new-tokens", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = _load_model(args.model_path, device)
    model.eval()

    dataset = build_dataset(
        args.dataset,
        args.subset,
        args.split,
        node_feat_dim=model.config.node_feat_dim,
    )
    dataset = filter_dataset(
        dataset,
        dataset_name=args.dataset,
        sample_idx=args.sample_idx,
    )
    if len(dataset) == 0:
        raise ValueError("No samples available after filtering.")

    sample = dataset[0]
    pyg_batch = create_pyg_batch(sample["graph"], device=device)

    wrapper = GLMWrapper(model, tokenizer)
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Initialize wrapper state.
    _ = wrapper.gen_output(sample["prompt"], pyg_batch, gen_cfg, num_trials=1)
    wrapper.set_generated_ids(sample["completion"])

    # Compute log-likelihood without and with relevant token filtering.
    with torch.no_grad():
        wrapper.relevant_idx = None
        ll_full = wrapper.forward(pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch)
        relevant_ids = wrapper.set_relevant_ids(args.baseline_graph, llr_threshold=3.0)
        ll_relevant = wrapper.forward(pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch)

    print(f"\n[RESULT] Full cumulative log-likelihood: {ll_full.item():.6f}")
    print(f"[RESULT] Relevant-only log-likelihood: {ll_relevant.item():.6f}")
    print(f"[RESULT] Relevant token IDs: {relevant_ids}")


if __name__ == "__main__":
    main()
