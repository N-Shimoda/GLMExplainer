import os

from datasets import concatenate_datasets

from eval import (
    _resolve_checkpoint_path,
    build_args,
    build_dataset,
    collect_result,
    eval_model,
    load_model_for_eval,
)

if __name__ == "__main__":
    args = build_args(multitask=True)
    model_path, run_name = _resolve_checkpoint_path(args.model_path)
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

        # Append accuracy to the log (file was initialized at start of run)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{subset}\t{args.split}\t{args.num_trials}\t{acc:.4f}\n")
