import argparse
import os

from datasets import concatenate_datasets
from eval import (
    _resolve_checkpoint_path,
    build_dataset,
    collect_result,
    eval_model,
    load_model_for_eval,
)


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--split", choices=["train", "validation", "test"], default="test")
    p.add_argument("--num_trials", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = build_args()
    model_path, run_name = _resolve_checkpoint_path(args.model_path)
    print(f"Checkpoint: {model_path}")
    print(f"Number of trials: {args.num_trials}")

    # Load model
    model = load_model_for_eval(model_path, load_llm_weights=False)
    model.eval()

    subsets = ["node_count", "edge_count", "cycle_check", "triangle_counting"]
    for subset in subsets:
        # Load dataset
        test_ds = build_dataset(subset, args.split, model.config.node_feat_dim)
        repeated_ds = concatenate_datasets([test_ds] * args.num_trials)

        # Evaluate
        print(f"Evaluating subset: {subset}")
        results = eval_model(model, repeated_ds, args.batch_size, subset)

        # Save results
        out_dir = os.path.join("results", subset)
        os.makedirs(out_dir, exist_ok=True)
        file_name = f"{run_name}_{args.split}.json" if run_name else f"results_{args.split}.json"
        res_file = os.path.join(out_dir, file_name)
        acc = collect_result(results, res_file, subset)
        print(f"[SUMMARY] subset={subset} accuracy={acc:.4f}")
