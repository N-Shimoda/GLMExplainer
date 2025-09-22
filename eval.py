import argparse
import json
import os
from math import ceil
from pprint import pprint  # noqa F401

import torch
from datasets import load_dataset
from torch_geometric.data import Batch as PygBatch
from torch_geometric.data import Data as PygData
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig

from src.glm import GraphTokenLM
from src.metrics import comp_accuracy
from src.preprocess import add_graph_column


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--subset",
        type=str,
        choices=["node_count", "edge_count", "cycle_check", "triangle_counting", "maximum_flow"],
        default="edge_count",
    )
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--num_graph_tokens", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def _checkpoint_step(path: str) -> int:
    name = os.path.basename(path.rstrip(os.sep))
    try:
        return int(name.split("-")[-1])
    except (ValueError, IndexError):
        return -1


def _resolve_checkpoint_path(model_path: str) -> str:
    """Resolve the concrete checkpoint directory to load."""
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        if os.path.isfile(config_path):
            return model_path

        candidates = [
            os.path.join(model_path, entry)
            for entry in os.listdir(model_path)
            if entry.startswith("checkpoint-") and os.path.isdir(os.path.join(model_path, entry))
        ]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint-* directories found under '{model_path}'.")

        candidates.sort(key=lambda p: (_checkpoint_step(p), p))
        best = candidates[-1]
        if _checkpoint_step(best) < 0:
            raise FileNotFoundError(
                f"Could not infer the last checkpoint under '{model_path}'. Provide a direct checkpoint path."
            )
        return best

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint path '{model_path}' does not exist.")

    return model_path


def create_pyg_batch(graph_dicts: list[dict[str, list]], device: torch.device) -> PygBatch:
    data_list = [
        PygData(
            x=torch.tensor(d["x"], dtype=torch.float).to(device),
            edge_index=torch.tensor(d["edge_index"], dtype=torch.long).to(device),
        )
        for d in graph_dicts
    ]
    batch = PygBatch.from_data_list(data_list)
    return batch


@torch.no_grad()
def eval_model(model: GraphTokenLM, test_ds, batch_size: int):
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.config.llm_name)
    gen_cfg = GenerationConfig(max_new_tokens=8, do_sample=False)

    results = []
    num_batches = ceil(len(test_ds) / batch_size)

    for i in tqdm(range(num_batches)):
        batch = test_ds[i * batch_size : (i + 1) * batch_size]
        pyg_batch = create_pyg_batch(batch["graph"], model.device)
        batch["graph"] = pyg_batch

        input_ids = tokenizer(batch["task_description"], return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(**input_ids, graph=pyg_batch, generation_config=gen_cfg)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        res_dict_li = [
            {
                "question": batch["task_description"][i],
                "preds": pred.split("\nA: ")[-1],
                "answer": batch["answer"][i],
            }
            for i, pred in enumerate(decoded)
        ]
        results.extend(res_dict_li)

    return results


def collect_result(results: list[dict], res_file: str, subset: str):
    """
    Evaluates prediction results, prints accuracy, unknown predictions, and saves results to a file.

    Parameters
    ----------
    results : list of dict
        A list of dictionaries containing prediction results. Each dictionary should have keys "preds" and "answer".
    res_file : str
        Path to the file where the results will be saved.
    subset : str
        The subset name used for accuracy computation.
    """
    # Compute accuracy
    acc, unknowns = comp_accuracy([r["preds"] for r in results], [r["answer"] for r in results], subset)
    print(f"Accuracy: {acc * 100:.2f}%")
    if unknowns:
        print("Unknown predictions:")
        for pred in unknowns:
            print(f" - {pred}")

    # Save results to a file
    os.makedirs(os.path.dirname(res_file), exist_ok=True)
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {res_file}")

    return acc


if __name__ == "__main__":
    args = build_args()

    # Load pre-trained model
    ckpt_path = _resolve_checkpoint_path(args.model_path)
    print(f"Checkpoint: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphTokenLM.from_pretrained(ckpt_path).to(device)

    # Load dataset
    test_raw = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_test").select(range(32))
    test_ds = test_raw.map(
        lambda x: add_graph_column(x, k=model.config.node_feat_dim),
        desc="add_graph_column(test)",
        remove_columns=["question", "nnodes", "nedges", "algorithm", "text_encoding"],
    )
    print("Test dataset:\n", test_ds)

    results = eval_model(model, test_ds, args.batch_size)
    res_file = os.path.join("results", args.subset, "results.json")
    collect_result(results, res_file, args.subset)
