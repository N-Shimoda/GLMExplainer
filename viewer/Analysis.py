from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from viewer.Page import AppPage


class AnalysisPage(AppPage):
    def __init__(self) -> None:
        super().__init__()
        self.subset_path: Path | None = None
        self.run_history: pd.DataFrame | None = None
        self.metric_options = {"auroc": "AUROC", "auprc": "AUPRC", "f1": "F1 Score"}

        if "average_mode" not in st.session_state:
            st.session_state["average_mode"] = True

    @staticmethod
    def load_run_history(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        required = {"run_name", "avg_auroc", "avg_auprc", "avg_f1"}
        if not required.issubset(df.columns):
            return None
        return df

    def create_selections(self) -> None:
        # Select subset
        subset_dirs = self.list_dirs(self.base_dir)
        subset_names = [p.name for p in subset_dirs]
        if not subset_names:
            st.error(f"No subsets found under `{self.base_dir}`.")
            st.stop()
        with st.sidebar:
            st.header("Run selection")
            default_index = subset_names.index(self.subset) if self.subset in subset_names else 0
            subset = st.selectbox("Subset", subset_names, index=default_index)
        st.session_state["subset"] = subset
        self.subset_path = self.base_dir / subset

        # Load run history
        run_history_path = self.subset_path / "run_history.csv"
        self.run_history = self.load_run_history(run_history_path)
        if self.run_history is None:
            st.error(f"Missing or invalid run history: `{run_history_path}`")
            st.stop()

        # Select left / right runs
        run_names = self.run_history["run_name"].astype(str).tolist()
        if not run_names:
            st.warning("No runs available in run history.")
            st.stop()
        with st.sidebar:
            left_default = run_names.index(self.left_run_name) if self.left_run_name in run_names else 0
            right_default = run_names.index(self.right_run_name) if self.right_run_name in run_names else 0
            self.left_run_name = st.selectbox("Left run", run_names, index=left_default)
            self.right_run_name = st.selectbox("Right run", run_names, index=right_default)
            st.session_state["left_run_name"] = self.left_run_name
            st.session_state["right_run_name"] = self.right_run_name

        # Display settings
        with st.sidebar:
            st.header("Display settings")
            self.metric_key = st.radio(
                "Metric", options=list(self.metric_options.keys()), format_func=lambda x: self.metric_options[x]
            )
            st.session_state["average_mode"] = st.toggle("Average per sample", value=st.session_state["average_mode"])

    def display_comparison(self) -> None:
        if self.run_history is None or self.left_run_name is None or self.right_run_name is None:
            st.error("Selections are incomplete.")
            st.stop()

        left_row = self.run_history[self.run_history["run_name"] == self.left_run_name]
        right_row = self.run_history[self.run_history["run_name"] == self.right_run_name]
        if left_row.empty or right_row.empty:
            st.warning("Selected runs are missing from history.")
            st.stop()

        avg_key = f"avg_{self.metric_key}"
        left_avg = float(left_row[avg_key].iloc[0])
        right_avg = float(right_row[avg_key].iloc[0])

        left_col, right_col = st.columns(2)
        with left_col:
            st.metric(self.metric_options[self.metric_key], value=f"{left_avg:.4f}")
        with right_col:
            st.metric(
                self.metric_options[self.metric_key],
                value=f"{right_avg:.4f}",
                delta=f"{right_avg - left_avg:+.4f}",
            )

        if self.subset_path is None:
            st.error("Subset is unavailable for per-trial comparison.")
            st.stop()

        left_samples_path = self.subset_path / self.left_run_name / "sample_metrics.csv"
        right_samples_path = self.subset_path / self.right_run_name / "sample_metrics.csv"
        required = {"sample_index", "trial", self.metric_key}
        left_samples = self.load_sample_metrics(left_samples_path, required=required)
        right_samples = self.load_sample_metrics(right_samples_path, required=required)
        if left_samples is None or right_samples is None:
            st.warning("Sample metrics are unavailable for one or both runs.")
            return

        left_metric = f"left_{self.metric_key}"
        right_metric = f"right_{self.metric_key}"

        # Prepare merged dataframe
        if st.session_state["average_mode"]:
            # Show average per sample across trials
            left_trimmed = (
                left_samples.groupby("sample_index", as_index=False)[self.metric_key]
                .mean()
                .rename(columns={self.metric_key: left_metric})
            )
            right_trimmed = (
                right_samples.groupby("sample_index", as_index=False)[self.metric_key]
                .mean()
                .rename(columns={self.metric_key: right_metric})
            )
            merged = left_trimmed.merge(right_trimmed, on="sample_index", how="inner")
            if merged.empty:
                st.warning("No overlapping samples found between the selected runs.")
                return
            merged["delta"] = merged[right_metric] - merged[left_metric]
            merged = merged.sort_values(["delta"], ascending=False)
            st.subheader(f"Average {self.metric_options[self.metric_key]} per sample")
        else:
            # Show per-trial metrics
            left_trimmed = left_samples[["sample_index", "trial", self.metric_key]].rename(
                columns={self.metric_key: left_metric}
            )
            right_trimmed = right_samples[["sample_index", "trial", self.metric_key]].rename(
                columns={self.metric_key: right_metric}
            )
            merged = left_trimmed.merge(right_trimmed, on=["sample_index", "trial"], how="inner")
            if merged.empty:
                st.warning("No overlapping trials found between the selected runs.")
                return
            merged["delta"] = merged[right_metric] - merged[left_metric]
            merged = merged.sort_values(["sample_index", "trial"])
            st.subheader(f"Per-trial {self.metric_options[self.metric_key]} comparison")

        column_config = {
            "sample_index": st.column_config.NumberColumn(width="small"),
            left_metric: st.column_config.NumberColumn(format="%.4f"),
            right_metric: st.column_config.NumberColumn(format="%.4f"),
            "delta": st.column_config.ProgressColumn(
                "Delta", min_value=-1.0, max_value=1.0, format="%.4f", color="auto"
            ),
        }
        if "trial" in merged.columns:
            column_config["trial"] = st.column_config.NumberColumn(width="small")

        # Display selectable dataframe
        selection = st.dataframe(
            merged,
            width="stretch",
            height=640,
            hide_index=True,
            column_config=column_config,
            on_select="rerun",
            selection_mode="single-row",
        )

        # Switch to Graph Viewer when a row is selected
        selected = selection["selection"]
        if selected and selected["rows"]:
            row = merged.iloc[selected["rows"][0]]
            st.session_state["graph_index"] = int(row["sample_index"])
            if "trial" in merged.columns:
                st.session_state["trial_index"] = int(row["trial"])
            else:
                st.session_state.pop("trial_index", None)
            st.switch_page("viewer/GraphViewer.py")

    def run(self) -> None:
        self.create_selections()
        self.display_comparison()


if __name__ == "__main__":
    st.set_page_config(page_title="AUROC Comparison", layout="wide")
    st.title("AUROC Comparison")

    viewer = AnalysisPage()
    viewer.run()
