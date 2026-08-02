"""Persistent settings for the viewer app.

Settings live in ``.streamlit/viewer.toml`` so that a directory picked on the
Config page survives restarts of ``app.py``. Streamlit reserves
``.streamlit/config.toml`` for its own options -- unknown keys there are only
warned about and are unreadable through ``st.get_option`` -- so the viewer keeps
its settings in a separate file next to it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

# `toml` is a dependency of Streamlit; `tomllib` is read-only, hence both.
import toml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / ".streamlit" / "viewer.toml"
DEFAULT_BASE_DIR = PROJECT_ROOT / "explanations"
SECTION = "explanations"
BASE_DIR_KEY = "dir"
RECENT_DIRS_KEY = "recent_dirs"
MAX_RECENT_DIRS = 5


def load_settings() -> dict:
    """Return the ``[explanations]`` table, or an empty dict if missing or broken."""
    section = _load_document().get(SECTION)
    return dict(section) if isinstance(section, dict) else {}


def save_settings(settings: dict) -> None:
    """Write the ``[explanations]`` table, leaving any other table untouched."""
    document = _load_document()
    document[SECTION] = settings
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(toml.dumps(document), encoding="utf-8")


def resolve_path(value: str | Path) -> Path:
    """Expand a user-supplied path (``~`` and project-relative paths are allowed)."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return Path(path)


def get_base_dir() -> Path:
    """Return the explanations directory the viewer should read from."""
    raw = load_settings().get(BASE_DIR_KEY)
    if not isinstance(raw, str) or not raw.strip():
        return DEFAULT_BASE_DIR
    return resolve_path(raw)


def get_raw_base_dir() -> str:
    """Return the configured directory as typed by the user, or the default as text."""
    raw = load_settings().get(BASE_DIR_KEY)
    if not isinstance(raw, str) or not raw.strip():
        return str(DEFAULT_BASE_DIR)
    return raw


def set_base_dir(value: str | Path) -> Path:
    """Persist ``value`` as the explanations directory and return the resolved path."""
    raw = str(value).strip()
    settings = load_settings()
    settings[BASE_DIR_KEY] = raw
    settings[RECENT_DIRS_KEY] = _push_recent(settings.get(RECENT_DIRS_KEY), raw)
    save_settings(settings)
    return resolve_path(raw)


def clear_base_dir() -> None:
    """Forget the custom directory so that the default one is used again."""
    settings = load_settings()
    settings.pop(BASE_DIR_KEY, None)
    save_settings(settings)


def get_recent_dirs() -> list[str]:
    """Return recently saved directories, most recent first."""
    recent = load_settings().get(RECENT_DIRS_KEY)
    if not isinstance(recent, list):
        return []
    return [entry for entry in recent if isinstance(entry, str) and entry.strip()]


def _load_document() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open("rb") as file:
            document = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return document if isinstance(document, dict) else {}


def _push_recent(recent: object, raw: str) -> list[str]:
    entries = [entry for entry in recent if isinstance(entry, str)] if isinstance(recent, list) else []
    return [raw, *[entry for entry in entries if entry != raw]][:MAX_RECENT_DIRS]
