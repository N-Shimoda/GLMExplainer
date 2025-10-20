import os
from typing import Literal


def _checkpoint_step(path: str) -> int:
    name = os.path.basename(path.rstrip(os.sep))
    try:
        return int(name.split("-")[-1])
    except (ValueError, IndexError):
        return -1


def _get_dir_type(path: str) -> Literal["task", "model", "checkpoint"]:
    """Classify directory type."""
    dir_name = os.path.basename(path.rstrip(os.sep))

    if dir_name in ["node_count", "edge_count", "cycle_check", "triangle_counting", "maximum_flow", "multitask"]:
        return "task"
    elif any(
        entry.startswith("checkpoint") and os.path.isdir(os.path.join(path, entry)) for entry in os.listdir(path)
    ):
        return "model"
    elif dir_name.startswith("checkpoint-"):
        return "checkpoint"
    else:
        raise ValueError(f"Directory '{path}' is neither a task, model, nor checkpoint directory.")


def _resolve_ckpt_path(model_path: str, version_index: int = -1) -> tuple[str, str]:
    """
    Resolve the concrete checkpoint directory to load.

    Parameters
    ----------
    model_path : str
        Path to the task directory, model directory or a specific checkpoint.
    version_index : int
        If multiple checkpoints exist, select the one with this index (0-based).
        -1 specifies the latest, 0 the earliest, etc.
        - The index should be specified within the range of available checkpoints.
        - This parameter is only used when `model_path` points to a task directory.

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
            latest_dir = os.path.join(model_path, run_dirs[version_index])
            return _resolve_ckpt_path(latest_dir)
        case "model":
            # This directory already contains the model (config.json present)
            run_name = os.path.basename(os.path.dirname(model_path.rstrip(os.sep)))
            checkpoint_dirs = [
                os.path.join(model_path, entry)
                for entry in os.listdir(model_path)
                if os.path.isdir(os.path.join(model_path, entry)) and entry.startswith("checkpoint")
            ]
            latest_ckpt = max(checkpoint_dirs, key=lambda p: (_checkpoint_step(p), p))
            return latest_ckpt, run_name
        case "checkpoint":
            # Final checkpoint directory (may or may not contain a config.json depending on layout)
            run_name = os.path.basename(os.path.dirname(model_path.rstrip(os.sep)))
            return model_path, run_name
        case _:
            # Fallback (should not reach here due to Literal constraint)
            return model_path, ""
