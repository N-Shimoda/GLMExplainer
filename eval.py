import argparse
import json
import os
from math import ceil

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Batch as PygBatch
from torch_geometric.data import Data as PygData
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, set_seed

from src.ckpt import _resolve_ckpt_path
from src.constants import GRAPHQA_SUBSETS, MOTIFQA_SUBSETS
from src.glm import GraphTokenLM
from src.metrics import comp_accuracy
from src.preprocess import add_graph_column

MAX_NEW_TOKENS = {
    "node_count": 4,
    "edge_count": 4,
    "cycle_check": 8,
    "triangle_counting": 4,
    "reachability": 4,
    "node_degree": 4,
    "edge_existence": 4,
    "ba_shapes": 8,
    "tree_cycle": 8,
    "tree_grid": 8,
    "tree_grid_v2": 8,
    "ba_two_motifs": 12,
    "shortest_path": 12,
}
EXT_MAX_NEW_TOKENS = {
    "node_count": 96,
    "edge_count": 256,
    "cycle_check": 512,
    "triangle_counting": 256,
}


def build_args():
    p = argparse.ArgumentParser()

    # Dataset settings
    p.add_argument("--dataset", type=str, choices=["GraphQA", "MotifQA"], default="MotifQA")
    p.add_argument(
        "--subset",
        type=str,
        nargs="+",
        choices=GRAPHQA_SUBSETS + MOTIFQA_SUBSETS,
        required=True,
        help="One or more subsets to evaluate.",
    )
    p.add_argument("--split", choices=["train", "validation", "test"], default="test")
    p.add_argument("--use-custom-dataset", action="store_true", default=False)

    # Model selection
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--model-index", type=int, default=-1, help="Which trained model version to use.")
    p.add_argument("--ckpt-index", type=int, default=-1, help="Which checkpoint version to use.")

    # Evaluation settings
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int)

    args = p.parse_args()
    _validate_args(args)

    return args


def _validate_args(args: argparse.Namespace):
    valid_subsets = GRAPHQA_SUBSETS if args.dataset == "GraphQA" else MOTIFQA_SUBSETS
    invalid = [subset for subset in args.subset if subset not in valid_subsets]
    if invalid:
        raise ValueError(f"Subsets {invalid} are not valid for dataset {args.dataset}.")
    if args.use_custom_dataset and args.dataset != "GraphQA":
        raise ValueError("--use-custom-dataset is only supported with GraphQA dataset.")


def load_model_for_eval(
    model_path: str, *, load_llm_weights: bool = False, device: torch.device | None = None
) -> GraphTokenLM:
    """Load GraphTokenLM onto a single device (DDP handles data-parallel)."""
    model = GraphTokenLM.from_pretrained(model_path, load_llm_weights=load_llm_weights)
    if device is not None:
        return model.to(device)
    if torch.cuda.is_available():
        return model.to("cuda:0")
    return model


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _infer_model_device(model: torch.nn.Module) -> torch.device:
    base_model = _unwrap_model(model)
    return next(base_model.parameters()).device


def _init_distributed() -> tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def _shard_dataset(total: int, rank: int, world_size: int) -> tuple[int, int]:
    per_rank = ceil(total / world_size) if world_size > 0 else total
    start = rank * per_rank
    end = min(start + per_rank, total)
    return start, end


def create_pyg_batch(
    graph_dicts: list[dict[str, list]] | dict[str, list], device: torch.device | str | None
) -> PygBatch:
    if isinstance(graph_dicts, dict):
        graph_dicts = [graph_dicts]

    def _as_tensor(value, *, dtype: torch.dtype) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.detach().clone().to(dtype=dtype)
        return torch.tensor(value, dtype=dtype)

    data_list = [
        PygData(
            x=_as_tensor(d["x"], dtype=torch.float),
            edge_index=_as_tensor(d["edge_index"], dtype=torch.long),
        )
        for d in graph_dicts
    ]
    batch = PygBatch.from_data_list(data_list)
    if device is not None:
        batch = batch.to(device)
    return batch


def build_dataset(
    dataset: str,
    subset: str,
    split: str,
    lpe_dim: int,
    use_degree_emb: bool = False,
):
    match dataset:
        case "GraphQA":
            test_raw = load_dataset("baharef/GraphQA", subset, split=f"zero_shot_{split}")
            test_ds = test_raw.map(
                lambda x: add_graph_column(
                    x,
                    ds_name="GraphQA",
                    lpe_dim=lpe_dim,
                    use_degree_emb=use_degree_emb,
                ),
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
        case "MotifQA":
            test_raw = load_dataset("naos-ku/motif-qa", subset, split=split)
            test_ds = test_raw.map(
                lambda x: add_graph_column(
                    x,
                    ds_name="MotifQA",
                    lpe_dim=lpe_dim,
                    use_degree_emb=use_degree_emb,
                ),
                remove_columns=["response", "nodes", "edges", "nnodes", "nedges"],
                desc="add_graph_column(test)",
            )
    return test_ds


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
    base_model = _unwrap_model(model)
    tokenizer = AutoTokenizer.from_pretrained(base_model.config.base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"

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
        outputs = base_model.generate(**input_ids, graph=pyg_batch, generation_config=gen_cfg)
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
    refs = [r["answer"] for r in results]
    preds = [r["preds"] for r in results]
    acc, unknowns, correct_mask = comp_accuracy(preds, refs, subset)
    for result, is_correct in zip(results, correct_mask):
        result.update({"correct": is_correct})
    print(f"[INFO] Accuracy: {acc * 100:.4f}%")
    if unknowns:
        print(f"[WARNING] {unknowns} unknown predictions found.")

    # Save results to a file
    os.makedirs(os.path.dirname(res_file), exist_ok=True)
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {res_file}")

    return acc


def main():
    use_dist, rank, world_size, local_rank = _init_distributed()
    is_main = rank == 0
    set_seed(42 + rank)
    args = build_args()

    # Load pre-trained model
    ckpt_path, run_name = _resolve_ckpt_path(args.model_path, args.model_index, args.ckpt_index)
    if is_main:
        print(f"Checkpoint: {ckpt_path}")
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model_for_eval(ckpt_path, load_llm_weights=False, device=device)
    if use_dist:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    base_model = _unwrap_model(model)

    lpe_dim = getattr(base_model.config, "lpe_dim", base_model.config.node_feat_dim)
    use_degree_emb = getattr(base_model.config, "use_degree_emb", False)
    for subset in args.subset:
        # Load dataset
        test_ds = build_dataset(
            args.dataset,
            subset,
            args.split,
            lpe_dim,
            use_degree_emb=use_degree_emb,
        )

        # Evaluate the model
        if args.max_new_tokens is not None:
            max_new_tokens = args.max_new_tokens
            if is_main:
                print(f"Using user-specified max_new_tokens: {max_new_tokens}")
        else:
            max_new_tokens_dict = EXT_MAX_NEW_TOKENS if args.use_custom_dataset else MAX_NEW_TOKENS
            max_new_tokens = max_new_tokens_dict.get(subset, 32)
        all_results = []
        for _ in range(args.num_trials):
            local_ds = test_ds
            if use_dist and world_size > 1:
                start, end = _shard_dataset(len(test_ds), rank, world_size)
                local_ds = test_ds.select(range(start, end))
            local_results = eval_model(model, local_ds, args.batch_size, max_new_tokens)

            # Save results
            if use_dist:
                gathered: list[list[dict]] = [None for _ in range(world_size)]
                dist.all_gather_object(gathered, local_results)
                if is_main:
                    all_results.extend([item for sublist in gathered for item in sublist])
            else:
                all_results.extend(local_results)

        if is_main:
            match args.split:
                case "test":
                    file_name = f"{run_name}.json" if run_name else "results.json"
                case _:
                    file_name = f"{run_name}_{args.split}.json" if run_name else f"results_{args.split}.json"
            res_file = os.path.join("results", subset, file_name)
            acc = collect_result(all_results, res_file, subset)
            print(f"[SUMMARY] subset={subset} accuracy={acc}")

    if use_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
