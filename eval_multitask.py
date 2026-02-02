import argparse
import os

from datasets import concatenate_datasets

from eval import (
    EXT_MAX_NEW_TOKENS,
    MAX_NEW_TOKENS,
    build_dataset,
    collect_result,
    eval_model,
    load_model_for_eval,
)
from src.ckpt import _resolve_ckpt_path
from src.constants import GRAPHQA_SUBSETS, MOTIFQA_SUBSETS


def _build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["GraphQA", "MotifQA"], required=True)
    parser.add_argument(
        "--subset",
        type=str,
        nargs="+",
        choices=GRAPHQA_SUBSETS + MOTIFQA_SUBSETS,
        required=True,
        help="One or more subsets to evaluate.",
    )
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--use-custom-dataset", action="store_true", default=False)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-index", type=int, default=-1, help="Which trained model version to use.")
    parser.add_argument("--ckpt-index", type=int, default=-1, help="Which checkpoint version to use.")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace):
    valid_subsets = GRAPHQA_SUBSETS if args.dataset == "GraphQA" else MOTIFQA_SUBSETS
    invalid = [subset for subset in args.subset if subset not in valid_subsets]
    if invalid:
        raise ValueError(f"Subsets {invalid} are not valid for dataset {args.dataset}.")
    if args.use_custom_dataset and args.dataset != "GraphQA":
        raise ValueError("--use-custom-dataset is only supported with GraphQA dataset.")


if __name__ == "__main__":
    args = _build_args()
    _validate_args(args)
    model_path, run_name = _resolve_ckpt_path(args.model_path, args.model_index, args.ckpt_index)
    print(f"Checkpoint: {model_path}")
    print(f"Number of trials: {args.num_trials}")

    # Logging setup: create (or overwrite) a new log file for each run
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    if run_name:
        log_path = os.path.join(log_dir, f"eval_multitask_{run_name}.log")
    else:
        log_path = os.path.join(log_dir, "eval_multitask.log")
    # Initialize file with header row
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("subset\tsplit\ttrials\taccuracy\n")

    # Load model
    model = load_model_for_eval(model_path, load_llm_weights=False)
    lpe_dim = getattr(model.config, "lpe_dim", model.config.node_feat_dim)
    use_degree_emb = getattr(model.config, "use_degree_emb", False)

    for subset in args.subset:
        # Load dataset
        test_ds = build_dataset(
            args.dataset,
            subset,
            args.split,
            lpe_dim,
            use_degree_emb=use_degree_emb,
        )
        repeated_ds = concatenate_datasets([test_ds] * args.num_trials)

        # Evaluate
        print(f"Evaluating subset: {subset}")
        if args.max_new_tokens is not None:
            max_new_tokens = args.max_new_tokens
        else:
            max_new_tokens_dict = EXT_MAX_NEW_TOKENS if args.use_custom_dataset else MAX_NEW_TOKENS
            max_new_tokens = max_new_tokens_dict.get(subset, 32)
        results = eval_model(model, repeated_ds, args.batch_size, max_new_tokens)

        # Save results
        out_dir = os.path.join("results", subset)
        os.makedirs(out_dir, exist_ok=True)
        file_name = f"{run_name}_{args.split}.json" if run_name else f"results_{args.split}.json"
        res_file = os.path.join(out_dir, file_name)
        acc = collect_result(results, res_file, subset)
        print(f"[SUMMARY] subset={subset} accuracy={acc:.4f}")

        # Append accuracy to the log (file was initialized at start of run)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{subset}\t{args.split}\t{args.num_trials}\t{acc:.4f}\n")
