from __future__ import annotations

import base64
import random
import re
from pathlib import Path

import streamlit as st

from viewer.Page import AppPage


class GraphViewerPage(AppPage):
    def __init__(self) -> None:
        super().__init__()
        self.subset_path: Path | None = None
        self.left_pdf_name: str | None = None
        self.right_pdf_name: str | None = None
        self.trial: int | None = None
        self.left_has_graph = False
        self.right_has_graph = False
        self.graph_index = st.session_state.get("graph_index", 0)

    @staticmethod
    def list_graph_files(path: Path) -> list[Path]:
        if not path.exists():
            return []
        files = list(path.glob("*.pdf")) + list(path.glob("*.svg"))
        return sorted(files)

    @staticmethod
    def embed_graph(path: Path) -> None:
        if path.suffix.lower() == ".svg":
            svg_text = path.read_text(encoding="utf-8")
            container = st.container(border=True)
            with container:
                st.markdown(
                    f'<div style="width: 100%; height: 420px; overflow: auto;">{svg_text}</div>',
                    unsafe_allow_html=True,
                )
            return
        pdf_bytes = path.read_bytes()
        b64 = base64.b64encode(pdf_bytes).decode("ascii")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" width=100% height=420></iframe>',
            unsafe_allow_html=True,
        )

    @staticmethod
    def graph_sort_key(name: str) -> tuple[int, str]:
        match = re.match(r"graph_(\d+)$", name)
        if match:
            return (0, f"{int(match.group(1)):09d}")
        return (1, name)

    @staticmethod
    def graph_file_counter(name: str) -> int | None:
        match = re.match(r".*_(\d+)\.(pdf|svg)$", name)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def parse_graph_index(name: str) -> int | None:
        match = re.match(r"graph_(\d+)$", name)
        if match:
            return int(match.group(1))
        return None

    def create_selections(self) -> None:
        # Select subset
        subset_dirs = self.list_dirs(self.base_dir)
        subset_names = [p.name for p in subset_dirs]
        with st.sidebar:
            default_index = subset_names.index(self.subset) if self.subset in subset_names else 0
            subset = st.selectbox("Subset", subset_names, index=default_index, key="subset_name")
            st.session_state["subset"] = subset

        self.subset_path = self.base_dir / subset
        run_dirs = self.list_dirs(self.subset_path)
        run_names = [p.name for p in run_dirs]

        # Select left / right runs
        left_run, right_run = st.columns(2)
        with left_run:
            left_idx = run_names.index(self.left_run_name) if self.left_run_name in run_names else 0
            self.left_run_name = st.selectbox("Left run", run_names, index=left_idx)
            st.session_state["left_run_name"] = self.left_run_name
        with right_run:
            right_idx = run_names.index(self.right_run_name) if self.right_run_name in run_names else 0
            self.right_run_name = st.selectbox("Right run", run_names, index=right_idx)
            st.session_state["right_run_name"] = self.right_run_name

        left_graphs = self.list_dirs(self.subset_path / self.left_run_name / "graphs")
        right_graphs = self.list_dirs(self.subset_path / self.right_run_name / "graphs")

        left_indices = [self.parse_graph_index(p.name) for p in left_graphs]
        right_indices = [self.parse_graph_index(p.name) for p in right_graphs]
        all_graph_indices = sorted(
            {idx for idx in left_indices + right_indices if idx is not None}
        )
        if not all_graph_indices:
            st.warning("No graphs found for the selected runs.")
            st.stop()

        # Random pick button
        pick_random = st.sidebar.button("Pick a graph", icon="🎲")
        if pick_random:
            st.session_state["graph_index"] = random.choice(all_graph_indices)
        self.graph_index = st.sidebar.selectbox("Graph", all_graph_indices, key="graph_index")

        left_graph_path = self.subset_path / self.left_run_name / "graphs" / Path(f"graph_{self.graph_index}")
        right_graph_path = self.subset_path / self.right_run_name / "graphs" / Path(f"graph_{self.graph_index}")
        self.left_has_graph = left_graph_path.exists()
        self.right_has_graph = right_graph_path.exists()

        left_files = self.list_graph_files(left_graph_path)
        right_files = self.list_graph_files(right_graph_path)

        left_pdf_by_counter = {}
        for name in (p.name for p in left_files):
            counter = self.graph_file_counter(name)
            if counter is not None and counter not in left_pdf_by_counter:
                left_pdf_by_counter[counter] = name

        right_pdf_by_counter = {}
        for name in (p.name for p in right_files):
            counter = self.graph_file_counter(name)
            if counter is not None and counter not in right_pdf_by_counter:
                right_pdf_by_counter[counter] = name

        if self.left_has_graph and self.right_has_graph:
            available_counters = sorted(set(left_pdf_by_counter) & set(right_pdf_by_counter))
            if not available_counters:
                st.warning("No common PDF files found in the selected graph.")
                st.stop()
        elif self.left_has_graph:
            available_counters = sorted(left_pdf_by_counter)
            if not available_counters:
                st.warning("No PDF files found in the selected left graph.")
                st.stop()
        else:
            available_counters = sorted(right_pdf_by_counter)
            if not available_counters:
                st.warning("No PDF files found in the selected right graph.")
                st.stop()

        if pick_random:
            st.session_state["trial_index"] = random.choice(available_counters)
        if "trial_index" not in st.session_state or st.session_state["trial_index"] not in available_counters:
            st.session_state["trial_index"] = available_counters[0]
        pdf_counter_value = st.sidebar.number_input(
            "Trial index",
            min_value=min(available_counters),
            max_value=max(available_counters),
            value=st.session_state["trial_index"],
            step=1,
            help=f"Available indices are {available_counters}",
            key="trial_index",
        )

        self.left_pdf_name = left_pdf_by_counter.get(pdf_counter_value)
        self.right_pdf_name = right_pdf_by_counter.get(pdf_counter_value)
        self.trial = int(pdf_counter_value)

    def create_graphs(self) -> None:
        left_graph_path = self.subset_path / self.left_run_name / "graphs" / Path(f"graph_{self.graph_index}")
        right_graph_path = self.subset_path / self.right_run_name / "graphs" / Path(f"graph_{self.graph_index}")

        left_pdf = left_graph_path / self.left_pdf_name if self.left_pdf_name else None
        right_pdf = right_graph_path / self.right_pdf_name if self.right_pdf_name else None

        left_metrics_path = self.subset_path / self.left_run_name / "sample_metrics.csv"
        right_metrics_path = self.subset_path / self.right_run_name / "sample_metrics.csv"
        left_metrics = (
            self._load_graph_metrics(left_metrics_path, self.graph_index, self.trial)
            if self.left_has_graph
            else None
        )
        right_metrics = (
            self._load_graph_metrics(right_metrics_path, self.graph_index, self.trial)
            if self.right_has_graph
            else None
        )

        left_col, right_col = st.columns(2)
        both_have_graph = self.left_has_graph and self.right_has_graph

        if self.left_has_graph and left_metrics is None:
            with left_col:
                st.warning("Metrics unavailable for the selected graph.")
        if self.right_has_graph and right_metrics is None:
            with right_col:
                st.warning("Metrics unavailable for the selected graph.")

        if both_have_graph and left_metrics is not None and right_metrics is not None:
            with left_col:
                self.display_graph(left_pdf, left_metrics, left_metrics, delta_color="off")
            with right_col:
                self.display_graph(right_pdf, right_metrics, left_metrics)
            return

        if self.left_has_graph and left_metrics is not None:
            with left_col:
                self.display_graph(left_pdf, left_metrics, left_metrics, delta_color="off")
        elif not self.left_has_graph:
            with left_col:
                st.info("Graph is not available in the left run.")

        if self.right_has_graph and right_metrics is not None:
            with right_col:
                self.display_graph(right_pdf, right_metrics, right_metrics, delta_color="off")
        elif not self.right_has_graph:
            with right_col:
                st.info("Graph is not available in the right run.")

    def display_graph(
        self,
        file_path: Path,
        metrics: dict[str, float],
        delta_base: dict[str, float],
        delta_color: str = "normal",
    ) -> None:
        metric_cols = st.columns(3)
        for col, key, label in zip(metric_cols, ["auroc", "auprc", "f1"], ["AUROC", "AUPRC", "F1"]):
            value = metrics[key]
            delta = value - delta_base[key]
            col.metric(label, value=f"{value:.4f}", delta=f"{delta:+.4f}", delta_color=delta_color)
        self.embed_graph(file_path)

    def _load_graph_metrics(self, path: Path, graph_index: int, trial: int) -> dict[str, float] | None:
        df = self.load_sample_metrics(path, required={"sample_index", "trial", "auroc", "auprc", "f1"})
        if df is None:
            return None
        filtered = df[(df["sample_index"] == graph_index) & (df["trial"] == trial)]
        if filtered.empty:
            return None
        row = filtered.iloc[0]
        return {
            "auroc": float(row["auroc"]),
            "auprc": float(row["auprc"]),
            "f1": float(row["f1"]),
        }

    def run(self) -> None:
        self.create_selections()
        self.create_graphs()


if __name__ == "__main__":
    st.set_page_config(page_title="Explanation Graph Viewer", layout="wide")
    st.title("Explanation Graph Viewer")

    viewer = GraphViewerPage()
    viewer.run()
