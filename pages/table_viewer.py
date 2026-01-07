from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from pages.utils import list_dirs, load_sample_metrics, selectbox_with_state

EXPLANATIONS_DIR = Path("explanations")


class AUROCComparisonViewer:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.subset_path: Path | None = None
        self.run_history: pd.DataFrame | None = None
        self.left_run_name: str | None = None
        self.right_run_name: str | None = None
        self.table_mode: str = "Per-trial"

    @staticmethod
    def load_run_history(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if "run_name" not in df.columns or "avg_auroc" not in df.columns:
            return None
        return df

    def create_selections(self) -> None:
        subset_dirs = list_dirs(self.base_dir)
        subset_names = [p.name for p in subset_dirs]
        if not subset_names:
            st.error(f"No subsets found under `{self.base_dir}`.")
            st.stop()
        subset = selectbox_with_state("Subset", subset_names, "subset_name", container=st.sidebar)
        self.subset_path = self.base_dir / subset

        run_history_path = self.subset_path / "run_history.csv"
        self.run_history = self.load_run_history(run_history_path)
        if self.run_history is None:
            st.error(f"Missing or invalid run history: `{run_history_path}`")
            st.stop()

        run_names = self.run_history["run_name"].astype(str).tolist()
        if not run_names:
            st.warning("No runs available in run history.")
            st.stop()

        left_col, right_col = st.columns(2)
        with left_col:
            self.left_run_name = selectbox_with_state("Left run", run_names, "left_run_name", container=left_col)
        with right_col:
            self.right_run_name = selectbox_with_state(
                "Right run",
                run_names,
                "right_run_name",
                default_index=min(1, len(run_names) - 1),
                container=right_col,
            )

        self.table_mode = st.sidebar.radio(
            "Table view",
            options=["Average per sample", "Per-trial"],
        )

    def display_comparison(self) -> None:
        if self.run_history is None or self.left_run_name is None or self.right_run_name is None:
            st.error("Selections are incomplete.")
            st.stop()

        left_row = self.run_history[self.run_history["run_name"] == self.left_run_name]
        right_row = self.run_history[self.run_history["run_name"] == self.right_run_name]
        if left_row.empty or right_row.empty:
            st.warning("Selected runs are missing from history.")
            st.stop()

        left_auroc = float(left_row["avg_auroc"].iloc[0])
        right_auroc = float(right_row["avg_auroc"].iloc[0])

        left_col, right_col = st.columns(2)
        with left_col:
            st.metric("AUROC", value=f"{left_auroc:.4f}")
        with right_col:
            st.metric("AUROC", value=f"{right_auroc:.4f}", delta=f"{right_auroc - left_auroc:+.4f}")

        if self.subset_path is None:
            st.error("Subset is unavailable for per-trial comparison.")
            st.stop()

        left_samples_path = self.subset_path / self.left_run_name / "sample_metrics.csv"
        right_samples_path = self.subset_path / self.right_run_name / "sample_metrics.csv"
        required = {"sample_index", "trial", "auroc"}
        left_samples = load_sample_metrics(left_samples_path, required=required)
        right_samples = load_sample_metrics(right_samples_path, required=required)
        if left_samples is None or right_samples is None:
            st.warning("Sample metrics are unavailable for one or both runs.")
            return

        match self.table_mode:
            case "Average per sample":
                left_trimmed = (
                    left_samples.groupby("sample_index", as_index=False)["auroc"]
                    .mean()
                    .rename(columns={"auroc": "left_auroc"})
                )
                right_trimmed = (
                    right_samples.groupby("sample_index", as_index=False)["auroc"]
                    .mean()
                    .rename(columns={"auroc": "right_auroc"})
                )
                merged = left_trimmed.merge(right_trimmed, on="sample_index", how="inner")
                if merged.empty:
                    st.warning("No overlapping samples found between the selected runs.")
                    return
                merged["delta"] = merged["right_auroc"] - merged["left_auroc"]
                merged = merged.sort_values(["sample_index"])
                st.subheader("Average AUROC per sample")
            case "Per-trial":
                left_trimmed = left_samples[["sample_index", "trial", "auroc"]].rename(columns={"auroc": "left_auroc"})
                right_trimmed = right_samples[["sample_index", "trial", "auroc"]].rename(
                    columns={"auroc": "right_auroc"}
                )
                merged = left_trimmed.merge(right_trimmed, on=["sample_index", "trial"], how="inner")
                if merged.empty:
                    st.warning("No overlapping trials found between the selected runs.")
                    return
                merged["delta"] = merged["right_auroc"] - merged["left_auroc"]
                merged = merged.sort_values(["sample_index", "trial"])
                st.subheader("Per-trial AUROC comparison")
        column_config = {
            "sample_index": st.column_config.NumberColumn(width="small"),
            "left_auroc": st.column_config.NumberColumn(format="%.4f"),
            "right_auroc": st.column_config.NumberColumn(format="%.4f"),
            "delta": st.column_config.ProgressColumn(
                "Delta", min_value=-1.0, max_value=1.0, format="%.4f", color="auto"
            ),
        }
        if "trial" in merged.columns:
            column_config["trial"] = st.column_config.NumberColumn(width="small")
        selection = st.dataframe(
            merged,
            width="stretch",
            hide_index=True,
            column_config=column_config,
            on_select="rerun",
            selection_mode="single-row",
        )
        selected = selection.get("selection") if isinstance(selection, dict) else getattr(selection, "selection", None)
        if selected and selected.get("rows"):
            if "trial" not in merged.columns:
                st.info("Switch to per-trial mode to open a graph for the selected sample.")
                return
            row = merged.iloc[selected["rows"][0]]
            st.session_state["graph_index"] = int(row["sample_index"])
            st.session_state["trial_index"] = int(row["trial"])
            st.switch_page("pages/graph_viewer.py")

    def run(self) -> None:
        self.create_selections()
        self.display_comparison()


if __name__ == "__main__":
    st.set_page_config(page_title="AUROC Comparison", layout="wide")
    st.title("AUROC Comparison")

    if not EXPLANATIONS_DIR.exists():
        st.error(f"Missing explanations directory: `{EXPLANATIONS_DIR}`")
        st.stop()

    viewer = AUROCComparisonViewer(EXPLANATIONS_DIR)
    viewer.run()
