import os
from typing import Literal, Tuple

from src.constants import GRAPHQA_SUBSETS, MOTIFQA_SUBSETS


def _get_dir_type(path: str) -> Literal["task", "model", "checkpoint"]:
    """Classify directory type."""
    dir_name = os.path.basename(path.rstrip(os.sep))
    if any(entry.startswith("checkpoint") and os.path.isdir(os.path.join(path, entry)) for entry in os.listdir(path)):
        return "model"
    elif dir_name.startswith("checkpoint-"):
        return "checkpoint"
    elif dir_name in GRAPHQA_SUBSETS + MOTIFQA_SUBSETS + ["multitask"]:
        return "task"
    else:
        raise ValueError(f"Directory '{path}' is neither a task, model, nor checkpoint directory.")


def _resolve_ckpt_path(model_path: str, model_index: int = -1, ckpt_index: int = -1) -> Tuple[str, str]:
    """
    Resolve the concrete checkpoint directory to load.

    Parameters
    ----------
    model_path : str
        Path to the task directory, model directory or a specific checkpoint.
    model_index : int
        If multiple checkpoints exist, select the one with this index (0-based).
        -1 specifies the latest, 0 the earliest, etc.
        - The index should be specified within the range of available checkpoints.
        - This parameter is only used when `model_path` points to a task directory.
    ckpt_index : int
        If multiple checkpoints exist within a model directory, select the one with this index (0-based).
        -1 specifies the latest, 0 the earliest, etc.
        - The index should be specified within the range of available checkpoints.
        - This parameter is only used when `model_path` points to a model or task directory.

    Returns
    -------
    ckpt_path : str
        Resolved checkpoint path.
    run_name : str
        Run directory name if applicable, else empty string.
    """
    # Explicit existence & directory checks (duplicated with _get_dir_type by intent)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Path '{model_path}' does not exist.")
    if not os.path.isdir(model_path):
        raise ValueError(f"'{model_path}' is not a directory.")

    # Determine directory type (still uses helper for classification)
    dir_type = _get_dir_type(model_path)

    match dir_type:
        case "task":
            # Select the latest run directory by numeric tuple ordering (e.g. '0-0', '0-1', ...)
            run_dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
            if not run_dirs:
                raise FileNotFoundError(f"No run directories found under '{model_path}'.")

            def _run_sort_key(name: str):
                parts = name.split("-")
                key = []
                for p in parts:
                    try:
                        key.append(int(p))
                    except ValueError:
                        key.append(-1)  # Non-int parts go first
                return key

            run_dirs.sort(key=_run_sort_key)
            latest_dir = os.path.join(model_path, run_dirs[model_index])
            return _resolve_ckpt_path(latest_dir, ckpt_index=ckpt_index)
        case "model":
            # This directory already contains the model (config.json present)
            run_name = os.path.basename(model_path.rstrip(os.sep))
            ckpt_dirs = [
                os.path.join(model_path, entry)
                for entry in os.listdir(model_path)
                if os.path.isdir(os.path.join(model_path, entry)) and entry.startswith("checkpoint")
            ]
            
            def _ckpt_sort_key(path: str):
                """Extract numeric checkpoint step from path like 'checkpoint-123'."""
                basename = os.path.basename(path)
                parts = basename.split("-")
                if len(parts) >= 2:
                    try:
                        return int(parts[-1])
                    except ValueError:
                        pass
                return 0  # Non-numeric or malformed checkpoints go first
            
            # Ensure deterministic ordering of checkpoints (e.g., for "latest"/"earliest" selection)
            ckpt_dirs.sort(key=_ckpt_sort_key)
            if not ckpt_dirs:
                raise FileNotFoundError(f"No checkpoint directories found under '{model_path}'.")
            
            try:
                ckpt_dir = ckpt_dirs[ckpt_index]
            except IndexError:
                raise IndexError(
                    f"Checkpoint index {ckpt_index} is out of range. "
                    f"Valid indices are 0 to {len(ckpt_dirs)-1} (or -1 to -{len(ckpt_dirs)}) "
                    f"for {len(ckpt_dirs)} checkpoints under '{model_path}'."
                )
            return ckpt_dir, run_name
        case "checkpoint":
            # Final checkpoint directory (may or may not contain a config.json depending on layout)
            if ckpt_index != -1:
                print("[WARNING] Ignored `checkpoint_index` because a specific checkpoint path is provided.")
            run_name = os.path.basename(os.path.dirname(model_path.rstrip(os.sep)))
            return model_path, run_name
        case _:
            raise RuntimeError(f"Unrecognized directory type for path '{model_path}'.")
