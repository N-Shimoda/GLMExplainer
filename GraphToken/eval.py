import argparse
import time

from datasets import load_dataset
from utils.utils import eval_model


def build_args():
    """
    Parses and returns command-line arguments for fine-tuning a Qwen3-4B model on the GraphQA dataset.

    Returns
    -------
    argparse.Namespace
        An object containing all the parsed command-line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="base model")
    p.add_argument(
        "--subset",
        type=str,
        required=True,
        help="GraphQA subset（see https://huggingface.co/datasets/baharef/GraphQA）",
    )
    p.add_argument("--model_path", type=str, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = build_args()

    # Build system instruction
    match args.subset:
        case "node_count":
            TASK_INST = "Answer ONLY with the final number of nodes."
        case "edge_count":
            TASK_INST = "Answer ONLY with the final number of edges."
        case "cycle_check":
            TASK_INST = "Answer ONLY with Yes or No."
        case _:
            raise NotImplementedError(f"Unsupported subset: {args.subset}")

    SYS_INST = "You are a careful graph reasoning assistant.\n" + TASK_INST

    # Dataset and model path
    test_ds = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_test")
    if args.model_path:
        model_path = args.model_path
        print(f"[INFO] Evaluating a fine-tuned model: {model_path}")
    else:
        model_path = args.model_name
        print(f"[INFO] Evaluating a pre-trained model: {model_path}")

    # Evaluate the model
    start_time = time.time()
    eval_model(model_path, test_ds, args.subset)
    print(f"[INFO] Evaluation completed in {time.time() - start_time:.2f} seconds")
