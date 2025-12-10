import argparse
import os

from datasets import arrow_dataset, load_dataset

from src.preprocess import add_graph_column


def build_dataset(dataset: str, subset: str, split: str, node_feat_dim: int) -> arrow_dataset.Dataset:
    """Build and return the specified dataset subset and split."""
    match dataset:
        case "GraphQA":
            ds_raw = load_dataset("baharef/GraphQA", subset, split=f"zero_shot_{split}")
            ds = ds_raw.map(
                lambda x: add_graph_column(x, k=node_feat_dim, ds_name="GraphQA"),
                remove_columns=["algorithm", "answer", "nedges", "nnodes", "task_description", "text_encoding"],
                load_from_cache_file=False,
            )
        case "MotifQA":
            ds_raw = load_dataset("naos-ku/motif-qa", "yes_no", split=split)
            ds = ds_raw.map(
                lambda x: add_graph_column(x, k=node_feat_dim, ds_name="MotifQA"),
                remove_columns=["response", "nedges", "nnodes"],
                load_from_cache_file=False,
            )
    return ds.add_column("index", list(range(len(ds))))


def filter_dataset(
    dataset: arrow_dataset.Dataset, args: argparse.Namespace, run_name: str
) -> tuple[arrow_dataset.Dataset, str]:
    """Filter dataset based on CLI args and return the filtered dataset and output directory."""
    subset = args.subset if args.subset is not None else "house_check"
    out_dir = os.path.join("explanations", subset, run_name)

    if args.sample_idx is not None:
        dataset = dataset.filter(lambda x: x["index"] == args.sample_idx)

    match args.dataset:
        case "MotifQA":
            if args.explain_pos_samples:
                dataset = dataset.filter(lambda x: len(x["motif_nodes"]) > 0)
                print("Extracted positive samples: len(dataset) =", len(dataset))
        case "GraphQA":
            if args.target_value is not None:
                dataset = dataset.filter(lambda x: int(x["completion"].split(".")[0]) == args.target_value)

    if args.num_samples is not None:
        num_to_select = min(args.num_samples, len(dataset))
        dataset = dataset.select(range(num_to_select))

    return dataset, out_dir
