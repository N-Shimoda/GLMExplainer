from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.constants import GRAPHQA_SUBSETS, MOTIFQA_SUBSETS
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

# Subset names `explain.py` writes under the explanations directory.
KNOWN_SUBSETS = frozenset(GRAPHQA_SUBSETS) | frozenset(MOTIFQA_SUBSETS)


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
        """Save a directory and drop the selections cached for the previous one."""
        raw = raw.strip()
        if not raw:
            st.toast("Directory is empty.", icon="⚠️")
            return
        set_base_dir(raw)
        cls.clear_dependent_state()
        st.toast(f"Saved: {resolve_path(raw)}", icon="✅")

    @classmethod
    def reset_directory(cls) -> None:
        """Fall back to the default directory in the project root."""
        clear_base_dir()
        cls.clear_dependent_state()
        st.toast("Reset to the default directory.", icon="↩️")

    @staticmethod
    def summarize_names(names: list[str], limit: int = 5) -> str:
        """Join names for a one-line message, trimming a long list."""
        shown = ", ".join(names[:limit])
        return shown if len(names) <= limit else f"{shown}, ... (+{len(names) - limit})"

    def unknown_subsets(self, path: Path) -> list[str]:
        """Return the subdirectory names of a path that are not known subsets."""
        return [name for name in self.list_dir_names(path) if name not in KNOWN_SUBSETS]

    def describe_status(self, path: Path) -> str:
        """Return a one-line badge describing whether a path is usable.

        Only a directory whose subdirectories are all known subset names counts
        as ready: any other name means the path is not an explanations root.
        """
        if not path.exists():
            return ":red-badge[:material/error: Missing] Directory does not exist."
        if not path.is_dir():
            return ":red-badge[:material/error: Invalid] Not a directory."
        names = self.list_dir_names(path)
        if not names:
            return ":orange-badge[:material/warning: Empty] No subset directory found."
        unknown = self.unknown_subsets(path)
        if unknown:
            return f":orange-badge[:material/warning: Unknown] Not a subset name: {self.summarize_names(unknown)}"
        return f":green-badge[:material/check: Ready] {len(names)} subset(s): {', '.join(names)}"

    def show_current(self) -> None:
        """Show the directory in use on a few lines, with a button to change it."""
        st.markdown("#### Explanations directory")
        current = get_base_dir()
        path_col, button_col = st.columns([5, 1], vertical_alignment="center")
        path_col.code(str(current), language="bash", wrap_lines=True)
        if button_col.button("Change", icon=":material/folder:", width="stretch"):
            self.open_editor()
        st.markdown(self.describe_status(current))

    def show_candidates(self, candidate: Path) -> None:
        """List the subdirectory names below the typed path as candidates to type next."""
        target = self.nearest_existing_dir(candidate)
        st.caption(f"Subdirectories of `{target}`")
        st.caption(str(self.list_dir_names(target)))

    def open_editor(self) -> None:
        """Open the editor dialog with the input primed with the saved directory."""
        # Safe to assign here: the input widget is only instantiated inside the dialog.
        st.session_state[INPUT_KEY] = get_raw_base_dir()
        self.show_editor()

    @st.dialog("Change directory", width="large")
    def show_editor(self) -> None:
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
            st.markdown(self.describe_status(candidate))

        unknown = self.unknown_subsets(candidate)
        save_col, reset_col, _ = st.columns([1, 1, 2])
        if save_col.button(
            "Save",
            type="primary",
            width="stretch",
            disabled=not candidate.is_dir() or not raw.strip() or bool(unknown),
            help="Every subdirectory has to be a subset name." if unknown else None,
        ):
            self.apply_directory(raw)
            st.rerun()
        if reset_col.button(
            "Reset to default",
            width="stretch",
            disabled=get_base_dir() == DEFAULT_BASE_DIR,
        ):
            self.reset_directory()
            st.rerun()

        self.show_recent()

    @staticmethod
    def fill_input(entry: str) -> None:
        """Put a recent directory into the input so its status can be read before saving.

        Written from a callback, which runs before the input widget is
        instantiated again; assigning to the key after that would be rejected.
        """
        st.session_state[INPUT_KEY] = entry

    def show_recent(self) -> None:
        recent = [entry for entry in get_recent_dirs() if entry != get_raw_base_dir()]
        if not recent:
            return
        st.divider()
        st.markdown("##### Recent directories")
        st.caption("Picking one fills the field above; press Save to apply it.")
        for entry in recent:
            path_col, button_col = st.columns([5, 1], vertical_alignment="center")
            path_col.code(entry, language="bash", wrap_lines=True)
            button_col.button(
                "Use",
                key=f"use_{entry}",
                width="stretch",
                disabled=entry == st.session_state.get(INPUT_KEY),
                on_click=self.fill_input,
                args=(entry,),
            )

    def run(self) -> None:
        self.show_current()
        st.caption(f"Settings are stored in `{CONFIG_PATH}`.")


if __name__ == "__main__":
    st.set_page_config(page_title="Config", layout="wide")
    st.title("Config")

    page = ConfigPage()
    page.run()
