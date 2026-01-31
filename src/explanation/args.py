import argparse

from src.constants import GRAPHQA_SUBSETS, MOTIFQA_SUBSETS


def check_non_negative_int(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"`{value}` is not an integer.")
    if ivalue < 0:
        raise argparse.ArgumentTypeError("Value must be non-negative.")
    return ivalue


def validate_args(args: argparse.Namespace) -> None:
    """Validates the parsed command-line arguments."""
    # Dataset and subset
    if args.dataset == "MotifQA" and args.subset not in MOTIFQA_SUBSETS:
        raise ValueError(f"Available MotifQA subsets are {MOTIFQA_SUBSETS} ({args.subset} was given).")
    if args.dataset == "GraphQA" and args.subset not in GRAPHQA_SUBSETS:
        raise ValueError(f"Available GraphQA subsets are {GRAPHQA_SUBSETS} ({args.subset} was given).")

    # Sample filtering
    if args.target_value is not None and args.sample_idx is not None:
        raise ValueError("Only one of `target_value` or `sample_idx` should be specified.")
    if args.target_pos_samples and args.dataset != "MotifQA":
        raise ValueError("`--target-pos-samples` is only supported for the MotifQA dataset.")
    if args.num_samples is not None and args.sample_idx is not None:
        raise ValueError("Only one of `num_samples` or `sample_idx` should be specified.")
    if args.num_trials < 1:
        raise ValueError("`num_trials` must be at least 1.")
    if args.min_correct_answers < 0:
        raise ValueError("`min_correct_answers` must be non-negative.")
    if args.min_correct_answers >= args.num_gen_trials:
        raise ValueError("`min_correct_answers` must be less than `num_gen_trials`.")
