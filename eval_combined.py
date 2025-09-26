import argparse
import os

import torch

from eval import _resolve_checkpoint_path, build_dataset, collect_result, eval_model
from src.glm import GraphTokenLM


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--split", choices=["train", "validation", "test"], default="test")
    return p.parse_args()


if __name__ == "__main__":
    args = build_args()
    model_path, run_name = _resolve_checkpoint_path(args.model_path)
    print(f"Checkpoint: {model_path}")
    print(f"Run name: {run_name if run_name else '(none)'}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GraphTokenLM.from_pretrained(model_path, load_llm_weights=False).to(device)

    subsets = ["node_count", "edge_count", "cycle_check", "triangle_counting", "maximum_flow"]
    for subset in subsets:
        # Load dataset
        test_ds = build_dataset(subset, args.split, model.config.node_feat_dim)

        # Evaluate
        print(f"Evaluating subset: {subset}")
        results = eval_model(model, test_ds, args.batch_size, subset)

        # Save results
        out_dir = os.path.join("results", subset)
        os.makedirs(out_dir, exist_ok=True)
        file_name = f"{run_name}_{args.split}.json" if run_name else f"results_{args.split}.json"
        res_file = os.path.join(out_dir, file_name)
        acc = collect_result(results, res_file, subset)
        print(f"[SUMMARY] subset={subset} accuracy={acc:.2f}")
