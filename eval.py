import argparse
import json
import os
from math import ceil

import torch
from datasets import concatenate_datasets, load_dataset
from torch_geometric.data import Batch as PygBatch
from torch_geometric.data import Data as PygData
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig

from src.constants import GRAPHQA_SUBSETS, MOTIFQA_SUBSETS
from src.ckpt import _resolve_ckpt_path
from src.glm import GraphTokenLM
from src.metrics import comp_accuracy
from src.preprocess import add_graph_column


def build_args(*, multitask: bool = False):
    p = argparse.ArgumentParser()

    # Dataset settings
    if not multitask:
        p.add_argument(
            "--subset",
            type=str,
            choices=GRAPHQA_SUBSETS + MOTIFQA_SUBSETS,
            default="edge_count",
        )
    p.add_argument("--use-custom-dataset", action="store_true", default=False)

    # Model selection
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--model-version-index", type=int, default=-1)

    # Evaluation settings
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--split", choices=["train", "validation", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int)

    return p.parse_args()


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


def create_pyg_batch(
    graph_dicts: list[dict[str, list]] | dict[str, list], device: torch.device | str | None
) -> PygBatch:
    if isinstance(graph_dicts, dict):
        graph_dicts = [graph_dicts]
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


def build_dataset(subset: str, split: str, node_feat_dim: int):
    match subset:
        case "node_count" | "edge_count" | "cycle_check" | "triangle_counting" | "reachability":
            test_raw = load_dataset("baharef/GraphQA", subset, split=f"zero_shot_{split}")
            test_ds = test_raw.map(
                lambda x: add_graph_column(x, k=node_feat_dim),
                desc="add_graph_column(test)",
                remove_columns=[
                    "algorithm",
                    "answer",
                    "nedges",
                    "nnodes",
                    "question",
                    "task_description",
                    "text_encoding",
                ],
            )
        case "house_check":
            test_raw = load_dataset("naos-ku/motif-qa", split=split)
            test_ds = test_raw.map(
                lambda x: add_graph_column(x, k=node_feat_dim, ds_name="MotifQA"),
                remove_columns=["response", "nodes", "edges", "nnodes", "nedges"],
                desc="add_graph_column(test)",
            )
    return test_ds


def get_max_new_tokens(subset: str, use_custom: bool = False) -> int:
    max_new_tokens_dict = (
        {"node_count": 96, "edge_count": 256, "cycle_check": 512, "triangle_counting": 256}
        if use_custom
        else {
            "node_count": 4,
            "edge_count": 4,
            "cycle_check": 8,
            "triangle_counting": 4,
            "reachability": 4,
            "house_check": 24,
        }
    )
    return max_new_tokens_dict.get(subset, 32)


@torch.no_grad()
def eval_model(model: GraphTokenLM, test_ds, batch_size: int, max_new_tokens: int) -> list[dict[str, str]]:
    """Evaluates the model on the test dataset and returns the prediction results.

    Parameters
    ----------
    model : GraphTokenLM
        The pre-trained GraphTokenLM model to be evaluated.
    test_ds : Dataset
        The test dataset containing prompts and graph data.
    batch_size : int
        The batch size for evaluation.
    max_new_tokens : int
        The maximum number of new tokens to generate.

    Returns
    -------
    results : list of dict
        A list of dictionaries containing the evaluation results with keys "question", "preds", and "answer".
    """
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.config.base_model)
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
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
    refs = [r["answer"] for r in results]
    acc, unknowns, _ = comp_accuracy([r["preds"] for r in results], refs, subset)
    print(f"Accuracy: {acc * 100:.4f}%")
    if unknowns:
        print(f"[WARNING] {unknowns} unknown predictions found.")

    # Save results to a file
    os.makedirs(os.path.dirname(res_file), exist_ok=True)
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {res_file}")

    return acc


def main():
    args = build_args()

    # Load pre-trained model
    ckpt_path, run_name = _resolve_ckpt_path(args.model_path, args.model_version_index)
    print(f"Checkpoint: {ckpt_path}")
    model = load_model_for_eval(ckpt_path, load_llm_weights=False)

    # Load dataset
    test_ds = build_dataset(args.subset, args.split, model.config.node_feat_dim)
    repeated_ds = concatenate_datasets([test_ds] * args.num_trials)

    # Evaluate the model
    if args.max_new_tokens is not None:
        max_new_tokens = args.max_new_tokens
        print(f"Using user-specified max_new_tokens: {max_new_tokens}")
    else:
        max_new_tokens = get_max_new_tokens(args.subset, args.use_custom_dataset)
    results = eval_model(model, repeated_ds, args.batch_size, max_new_tokens)

    # Save results
    match args.split:
        case "test":
            file_name = f"{run_name}.json" if run_name else "results.json"
        case _:
            file_name = f"{run_name}_{args.split}.json" if run_name else f"results_{args.split}.json"
    res_file = os.path.join("results", args.subset, file_name)
    acc = collect_result(results, res_file, args.subset)
    print(f"[SUMMARY] subset={args.subset} accuracy={acc}")


if __name__ == "__main__":
    main()
