import argparse
import json
import os
from math import ceil

import torch
from torch_geometric.data import Batch as PygBatch
from torch_geometric.data import Data as PygData
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig

from datasets import concatenate_datasets, load_dataset
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
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--split", choices=["train", "validation", "test"], default="test")
    p.add_argument("--num_trials", type=int, default=1)
    return p.parse_args()


def _checkpoint_step(path: str) -> int:
    name = os.path.basename(path.rstrip(os.sep))
    try:
        return int(name.split("-")[-1])
    except (ValueError, IndexError):
        return -1


def _resolve_checkpoint_path(model_path: str) -> tuple[str, str]:
    """
    Resolve the concrete checkpoint directory to load.

    Parameters
    ----------
    model_path : str
        Path to the task directory, model directory or a specific checkpoint.

    Returns
    -------
    ckpt_path : str
        Resolved checkpoint path.
    run_name : str
        Run directory name if applicable, else empty string.
    """
    if os.path.isdir(model_path):
        dir_name = os.path.basename(model_path.rstrip(os.sep))
        if dir_name in ["node_count", "edge_count", "cycle_check", "triangle_counting", "maximum_flow", "combined"]:
            run_dirs = os.listdir(model_path)
            if not run_dirs:
                raise FileNotFoundError(f"No run directories found under '{model_path}'.")
            run_dirs.sort(key=lambda p: (int(p.split("-")[0]), int(p.split("-")[1])))

            latest_dir = os.path.join(model_path, run_dirs[-1])
            return _resolve_checkpoint_path(latest_dir)

        config_path = os.path.join(model_path, "config.json")
        if os.path.isfile(config_path):
            run_name = os.path.basename(os.path.dirname(model_path.rstrip(os.sep)))
            return model_path, run_name

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
        return best, os.path.basename(model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint path '{model_path}' does not exist.")

    return model_path, ""


def load_model_for_eval(model_path: str, *, load_llm_weights: bool = False) -> GraphTokenLM:
    """Load GraphTokenLM with multi-GPU support when available."""
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            return GraphTokenLM.from_pretrained(
                model_path,
                load_llm_weights=load_llm_weights,
                device_map="auto",
            )
        return GraphTokenLM.from_pretrained(
            model_path,
            load_llm_weights=load_llm_weights,
        ).to("cuda")
    return GraphTokenLM.from_pretrained(model_path, load_llm_weights=load_llm_weights)


def build_dataset(subset: str, split: str, node_feat_dim: int):
    test_raw = load_dataset("baharef/GraphQA", subset, split=f"zero_shot_{split}")
    test_ds = test_raw.map(
        lambda x: add_graph_column(x, k=node_feat_dim),
        desc="add_graph_column(test)",
        remove_columns=["algorithm", "answer", "nedges", "nnodes", "question", "task_description", "text_encoding"],
    )
    return test_ds


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _infer_model_device(model: torch.nn.Module) -> torch.device:
    base_model = _unwrap_model(model)
    device_map = getattr(base_model, "hf_device_map", None)
    if device_map:
        skip_devices = {None, "cpu", "meta"}
        for device in device_map.values():
            if device in skip_devices:
                continue
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            if isinstance(device, torch.device):
                return device
            return torch.device(device)
        first_device = next(iter(device_map.values()))
        if isinstance(first_device, int):
            return torch.device(f"cuda:{first_device}")
        if isinstance(first_device, torch.device):
            return first_device
        if isinstance(first_device, str):
            return torch.device(first_device)
        # fall back to the module's first parameter device
        return next(base_model.parameters()).device
    return next(base_model.parameters()).device


def create_pyg_batch(graph_dicts: list[dict[str, list]], device: torch.device | str | None) -> PygBatch:
    data_list = [
        PygData(
            x=torch.tensor(d["x"], dtype=torch.float),
            edge_index=torch.tensor(d["edge_index"], dtype=torch.long),
        )
        for d in graph_dicts
    ]
    batch = PygBatch.from_data_list(data_list)
    if device is not None:
        batch = batch.to(device)
    return batch


@torch.no_grad()
def eval_model(model: GraphTokenLM, test_ds, batch_size: int, subset: str) -> list[dict]:
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.config.llm_name)

    max_new_token_dict = {"node_count": 3, "cycle_check": 8}
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_token_dict.get(subset, 6),
        do_sample=True,
    )

    results = []
    num_batches = ceil(len(test_ds) / batch_size)
    model_device = _infer_model_device(model)

    for i in tqdm(range(num_batches)):
        batch = test_ds[i * batch_size : (i + 1) * batch_size]
        pyg_batch = create_pyg_batch(batch["graph"], model_device)
        batch["graph"] = pyg_batch

        input_ids = tokenizer(batch["prompt"], return_tensors="pt", padding=True).to(model_device)
        outputs = model.generate(**input_ids, graph=pyg_batch, generation_config=gen_cfg)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        res_dict_li = [
            {
                "question": batch["prompt"][i],
                "preds": pred.split("\nA: ")[-1],
                "answer": batch["completion"][i],
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
    print(f"Accuracy: {acc * 100:.3f}%")
    if unknowns:
        print(f"[WARNING] {unknowns} unknown predictions found.")

    # Save results to a file
    os.makedirs(os.path.dirname(res_file), exist_ok=True)
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {res_file}")

    return acc


if __name__ == "__main__":
    args = build_args()

    # Load pre-trained model
    ckpt_path, run_name = _resolve_checkpoint_path(args.model_path)
    print(f"Checkpoint: {ckpt_path}")
    model = load_model_for_eval(ckpt_path, load_llm_weights=False)

    # Load dataset
    test_ds = build_dataset(args.subset, args.split, model.config.node_feat_dim)
    repeated_ds = concatenate_datasets([test_ds] * args.num_trials)
    print("Test dataset:\n", repeated_ds)

    results = eval_model(model, repeated_ds, args.batch_size, args.subset)
    match args.split:
        case "test":
            file_name = f"{run_name}.json" if run_name else "results.json"
        case _:
            file_name = f"{run_name}_{args.split}.json" if run_name else f"results_{args.split}.json"
    res_file = os.path.join("results", args.subset, file_name)
    acc = collect_result(results, res_file, args.subset)
    print(f"[SUMMARY] subset={args.subset} accuracy={acc:.3f}")
