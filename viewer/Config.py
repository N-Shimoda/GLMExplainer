from __future__ import annotations

from pathlib import Path

import streamlit as st

from viewer.settings import (
    CONFIG_PATH,
    DEFAULT_BASE_DIR,
    clear_base_dir,
    get_base_dir,
    get_raw_base_dir,
    get_recent_dirs,
    resolve_path,
    set_base_dir,
)

INPUT_KEY = "explanations_dir_input"


class ConfigPage:
    """Page for choosing which directory the viewer reads explanations from."""

    # Selections cached in the session that point into the previous directory.
    DEPENDENT_STATE_KEYS = (
        "subset",
        "subset_name",
        "left_run_name",
        "right_run_name",
        "graph_index",
        "trial_index",
    )

    @staticmethod
    def list_dir_names(path: Path) -> list[str]:
        """Return the visible subdirectory names directly under a path."""
        if not path.is_dir():
            return []
        try:
            return sorted(p.name for p in path.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError:
            return []

    @staticmethod
    def nearest_existing_dir(candidate: Path) -> Path:
        """Return the candidate itself, or its closest ancestor that exists.

        A path that is still being typed (``.../OneDr``) falls back to its parent,
        so candidate names keep showing up while the user types.
        """
        while not candidate.is_dir() and candidate != candidate.parent:
            candidate = candidate.parent
        return candidate

    @classmethod
    def clear_dependent_state(cls) -> None:
        """Drop cached selections so they are re-picked from the new directory."""
        for key in cls.DEPENDENT_STATE_KEYS:
            st.session_state.pop(key, None)

    @classmethod
    def apply_directory(cls, raw: str) -> None:
        """Save a directory and refresh both the input widget and cached selections."""
        raw = raw.strip()
        if not raw:
            st.toast("Directory is empty.", icon="⚠️")
            return
        set_base_dir(raw)
        cls.clear_dependent_state()
        st.session_state[INPUT_KEY] = raw
        st.toast(f"Saved: {resolve_path(raw)}", icon="✅")

    @classmethod
    def reset_directory(cls) -> None:
        """Fall back to the default directory in the project root."""
        clear_base_dir()
        cls.clear_dependent_state()
        st.session_state[INPUT_KEY] = str(DEFAULT_BASE_DIR)
        st.toast("Reset to the default directory.", icon="↩️")

    def show_status(self, path: Path) -> None:
        """Report whether a path is usable as an explanations directory."""
        if not path.exists():
            st.warning(f"Directory does not exist: `{path}`", icon="⚠️")
            return
        if not path.is_dir():
            st.warning(f"Not a directory: `{path}`", icon="⚠️")
            return
        subsets = self.list_dir_names(path)
        if not subsets:
            st.warning(f"No subset directory found under `{path}`.", icon="⚠️")
            return
        st.success(f"Found {len(subsets)} subset(s): {', '.join(subsets)}")

    def show_current(self) -> None:
        st.subheader("Current directory")
        current = get_base_dir()
        st.code(str(current), language="bash")
        self.show_status(current)

    def show_candidates(self, candidate: Path) -> None:
        """List the subdirectory names below the typed path as candidates to type next."""
        target = self.nearest_existing_dir(candidate)
        st.caption(f"Subdirectories of `{target}`")
        st.caption(str(self.list_dir_names(target)))

    def show_editor(self) -> None:
        st.subheader("Change directory")
        if INPUT_KEY not in st.session_state:
            st.session_state[INPUT_KEY] = get_raw_base_dir()
        raw = st.text_input(
            "Explanations directory",
            key=INPUT_KEY,
            help=(
                "Absolute path, `~`-relative path, or a path relative to the project root. "
                "For explanations synced to OneDrive, point this at the uploaded `explanations` folder."
            ),
        )
        candidate = resolve_path(raw) if raw.strip() else DEFAULT_BASE_DIR

        self.show_candidates(candidate)

        st.caption("Resolved path")
        st.code(str(candidate), language="bash")
        if candidate == get_base_dir():
            st.caption("This is the directory currently in use.")
        else:
            self.show_status(candidate)

        save_col, reset_col, _ = st.columns([1, 1, 3])
        save_col.button(
            "Save",
            type="primary",
            width="stretch",
            disabled=not candidate.is_dir() or not raw.strip(),
            on_click=self.apply_directory,
            args=(raw,),
        )
        reset_col.button(
            "Reset to default",
            width="stretch",
            disabled=get_base_dir() == DEFAULT_BASE_DIR,
            on_click=self.reset_directory,
        )

    def show_recent(self) -> None:
        recent = [entry for entry in get_recent_dirs() if entry != get_raw_base_dir()]
        if not recent:
            return
        st.subheader("Recent directories")
        for entry in recent:
            path_col, button_col = st.columns([5, 1], vertical_alignment="center")
            path_col.code(entry, language="bash")
            button_col.button(
                "Use",
                key=f"use_{entry}",
                width="stretch",
                on_click=self.apply_directory,
                args=(entry,),
            )

    def run(self) -> None:
        self.show_current()
        st.divider()
        self.show_editor()
        st.divider()
        self.show_recent()
        st.caption(f"Settings are stored in `{CONFIG_PATH}`.")


if __name__ == "__main__":
    st.set_page_config(page_title="Config", layout="wide")
    st.title("Config")

    page = ConfigPage()
    page.run()
