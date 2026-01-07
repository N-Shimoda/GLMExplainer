from __future__ import annotations

import base64
import random
import re
from pathlib import Path

import streamlit as st

from pages.utils import list_dirs, load_sample_metrics

EXPLANATIONS_DIR = Path("explanations")


class ExplanationGraphViewer:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.subset_path: Path | None = None
        self.left_run_name: str | None = None
        self.right_run_name: str | None = None
        self.left_pdf_name: str | None = None
        self.right_pdf_name: str | None = None
        self.graph_name: str | None = None
        self.trial: int | None = None

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
    def graph_index(name: str) -> int | None:
        match = re.match(r"graph_(\d+)$", name)
        if match:
            return int(match.group(1))
        return None

    def create_selections(self) -> None:
        subset_dirs = list_dirs(self.base_dir)
        subset_names = [p.name for p in subset_dirs]
        subset = st.sidebar.selectbox("Subset", subset_names)

        self.subset_path = self.base_dir / subset
        run_dirs = list_dirs(self.subset_path)
        run_names = [p.name for p in run_dirs]

        left_run, right_run = st.columns(2)
        with left_run:
            self.left_run_name = st.selectbox("Left run", run_names, index=0)
        with right_run:
            self.right_run_name = st.selectbox("Right run", run_names, index=min(1, len(run_names) - 1))

        left_graphs = list_dirs(self.subset_path / self.left_run_name / "graphs")
        right_graphs = list_dirs(self.subset_path / self.right_run_name / "graphs")

        common_graphs = sorted(
            {p.name for p in left_graphs} & {p.name for p in right_graphs},
            key=self.graph_sort_key,
        )

        if not common_graphs:
            st.warning("No common graphs found for the selected runs.")
            st.stop()

        if "graph_name" not in st.session_state or st.session_state["graph_name"] not in common_graphs:
            st.session_state["graph_name"] = common_graphs[0]
        pick_random = st.sidebar.button("Pick a graph", icon="🎲")
        if pick_random:
            st.session_state["graph_name"] = random.choice(common_graphs)
        self.graph_name = st.sidebar.selectbox("Graph", common_graphs, key="graph_name")

        left_graph_path = self.subset_path / self.left_run_name / "graphs" / self.graph_name
        right_graph_path = self.subset_path / self.right_run_name / "graphs" / self.graph_name

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

        common_counters = sorted(set(left_pdf_by_counter) & set(right_pdf_by_counter))

        if not common_counters:
            st.warning("No common PDF files found in the selected graph.")
            st.stop()

        if pick_random:
            st.session_state["trial_index"] = random.choice(common_counters)
        if "trial_index" not in st.session_state or st.session_state["trial_index"] not in common_counters:
            st.session_state["trial_index"] = common_counters[0]
        pdf_counter_value = st.sidebar.number_input(
            "Trial index",
            min_value=min(common_counters),
            max_value=max(common_counters),
            value=st.session_state["trial_index"],
            step=1,
            key="trial_index",
        )

        if pdf_counter_value not in left_pdf_by_counter or pdf_counter_value not in right_pdf_by_counter:
            st.warning("Selected PDF index is not available in both runs.")
            st.stop()

        self.left_pdf_name = left_pdf_by_counter[pdf_counter_value]
        self.right_pdf_name = right_pdf_by_counter[pdf_counter_value]
        self.trial = int(pdf_counter_value)

    def create_graphs(self) -> None:
        if (
            self.subset_path is None
            or self.left_run_name is None
            or self.right_run_name is None
            or self.left_pdf_name is None
            or self.right_pdf_name is None
            or self.graph_name is None
            or self.trial is None
        ):
            st.error("Selections are incomplete.")
            st.stop()

        left_graph_path = self.subset_path / self.left_run_name / "graphs" / self.graph_name
        right_graph_path = self.subset_path / self.right_run_name / "graphs" / self.graph_name

        left_pdf = left_graph_path / self.left_pdf_name
        right_pdf = right_graph_path / self.right_pdf_name

        graph_index = self.graph_index(self.graph_name)
        if graph_index is None:
            st.warning("Unable to extract graph index for metric comparison.")
            return
        left_metrics_path = self.subset_path / self.left_run_name / "sample_metrics.csv"
        right_metrics_path = self.subset_path / self.right_run_name / "sample_metrics.csv"
        left_metrics = self._load_graph_metrics(left_metrics_path, graph_index, self.trial)
        right_metrics = self._load_graph_metrics(right_metrics_path, graph_index, self.trial)
        if left_metrics is None or right_metrics is None:
            st.warning("Metrics unavailable for the selected graph.")
            return

        left_col, right_col = st.columns(2)
        with left_col:
            st.subheader(f"Left: `{self.left_run_name}`")
            self.display_graph(left_pdf, left_metrics, left_metrics, delta_color="off")
        with right_col:
            st.subheader(f"Right: `{self.right_run_name}`")
            self.display_graph(right_pdf, right_metrics, left_metrics)

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
        df = load_sample_metrics(path, required={"sample_index", "trial", "auroc", "auprc", "f1"})
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
        st.divider()
        self.create_graphs()


if __name__ == "__main__":
    st.title("Explanation Graph Viewer")
    st.set_page_config(page_title="Explanation Graph Viewer", layout="wide")

    if not EXPLANATIONS_DIR.exists():
        st.error(f"Missing explanations directory: `{EXPLANATIONS_DIR}`")
        st.stop()

    viewer = ExplanationGraphViewer(EXPLANATIONS_DIR)
    viewer.run()
