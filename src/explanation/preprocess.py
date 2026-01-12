from typing import Optional

from datasets import arrow_dataset, load_dataset

from src.preprocess import add_graph_column


def build_dataset(dataset: str, subset: str, split: str, node_feat_dim: int) -> arrow_dataset.Dataset:
    """Build and return the specified dataset subset and split."""
    match dataset:
        case "GraphQA":
            ds_raw = load_dataset("baharef/GraphQA", subset, split=f"zero_shot_{split}")
            ds = ds_raw.map(
                lambda x: add_graph_column(x, k=node_feat_dim, ds_name="GraphQA"),
                remove_columns=["algorithm", "answer", "nnodes", "nedges", "task_description", "text_encoding"],
                load_from_cache_file=False,
            )
        case "MotifQA":
            ds_raw = load_dataset("naos-ku/motif-qa", subset, split=split)
            ds = ds_raw.map(
                lambda x: add_graph_column(x, k=node_feat_dim, ds_name="MotifQA"),
                remove_columns=["response", "nnodes", "nedges"],
                load_from_cache_file=False,
            )
    return ds.add_column("index", list(range(len(ds))))


def filter_dataset(
    dataset: arrow_dataset.Dataset,
    dataset_name: str,
    target_pos_samples: bool = False,
    sample_idx: Optional[int] = None,
    target_value: Optional[int] = None,
    num_samples: Optional[int] = None,
) -> arrow_dataset.Dataset:
    """Filter dataset based on CLI args and return the filtered dataset."""
    if sample_idx is not None:
        dataset = dataset.filter(lambda x: x["index"] == sample_idx)

    match dataset_name:
        case "MotifQA":
            if target_pos_samples:
                dataset = dataset.filter(lambda x: len(x["motif_nodes"]) > 0)
                print("Extracted positive samples: len(dataset) =", len(dataset))
        case "GraphQA":
            if target_value is not None:
                dataset = dataset.filter(lambda x: int(x["completion"].split(".")[0]) == target_value)

    if num_samples is not None:
        num_to_select = min(num_samples, len(dataset))
        dataset = dataset.select(range(num_to_select))

    return dataset
