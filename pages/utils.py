from pathlib import Path

import pandas as pd


def list_dirs(path: Path) -> list[Path]:
    """Return sorted subdirectories for a path, or an empty list if missing."""
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir()])


def load_sample_metrics(path: Path, required: set[str] | None = None) -> pd.DataFrame | None:
    """Load a CSV file and optionally validate required columns."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if required and not required.issubset(df.columns):
        return None
    return df
