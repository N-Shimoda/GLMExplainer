import argparse
import os
from math import ceil

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
    p.add_argument("--num_graph_tokens", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=32)
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


if __name__ == "__main__":
    args = build_args()

    test_ds = load_dataset("baharef/GraphQA", args.subset, split="zero_shot_test")
    test_ds = test_ds.map(add_graph_column, desc="add_graph_column(test)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = _resolve_checkpoint_path(args.model_path)
    model = GraphTokenLM.from_pretrained(ckpt_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model.config.llm_name)

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        # eos_token_id=tokenizer.eos_token_id,
    )

    num_batches = ceil(len(test_ds) / args.batch_size)
    preds = []
    answers = []
    for i in tqdm(range(num_batches)):
        batch = test_ds[i * args.batch_size : (i + 1) * args.batch_size]
        pyg_batch = create_pyg_batch(batch["graph"], model.device)
        batch["graph"] = pyg_batch

        input_ids = tokenizer(batch["task_description"], return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**input_ids, graph=pyg_batch, generation_config=gen_cfg)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        preds.extend([pred.split("\nA: ")[-1] for pred in decoded])
        answers.extend(batch["answer"])

    acc, unknowns = comp_accuracy(preds, answers, args.subset)
    print(f"Accuracy: {acc * 100:.2f}%")
    if unknowns:
        print("Unknown predictions:")
        for pred in unknowns:
            print(f" - {pred}")
