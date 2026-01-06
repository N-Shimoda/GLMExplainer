from __future__ import annotations

import base64
import re
from pathlib import Path

import pandas as pd
import streamlit as st

EXPLANATIONS_DIR = Path("explanations")


class ExplanationGraphViewer:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    @staticmethod
    def list_dirs(path: Path) -> list[Path]:
        if not path.exists():
            return []
        return sorted([p for p in path.iterdir() if p.is_dir()])

    @staticmethod
    def list_pdfs(path: Path) -> list[Path]:
        if not path.exists():
            return []
        return sorted(path.glob("*.pdf"))

    @staticmethod
    def embed_pdf(path: Path) -> None:
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
    def pdf_counter(name: str) -> int | None:
        match = re.match(r".*_(\d+)\.pdf$", name)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def graph_index(name: str) -> int | None:
        match = re.match(r"graph_(\d+)$", name)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def load_sample_metrics(path: Path, graph_index: int) -> dict[str, float] | None:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if "sample_index" not in df.columns:
            return None
        filtered = df[df["sample_index"] == graph_index]
        if filtered.empty:
            return None
        return {
            "auroc": float(filtered["auroc"].mean()),
            "auprc": float(filtered["auprc"].mean()),
            "f1": float(filtered["f1"].mean()),
        }

    def create_selections(self) -> tuple[Path, str, str, str, str, str]:
        subset_dirs = self.list_dirs(self.base_dir)
        subset_names = [p.name for p in subset_dirs]
        subset = st.selectbox("Subset", subset_names)

        subset_path = self.base_dir / subset
        run_dirs = self.list_dirs(subset_path)
        run_names = [p.name for p in run_dirs]

        left_run, right_run = st.columns(2)
        with left_run:
            left_run_name = st.selectbox("Left run", run_names, index=0)
        with right_run:
            right_run_name = st.selectbox("Right run", run_names, index=min(1, len(run_names) - 1))

        left_graphs = self.list_dirs(subset_path / left_run_name / "graphs")
        right_graphs = self.list_dirs(subset_path / right_run_name / "graphs")

        common_graphs = sorted(
            {p.name for p in left_graphs} & {p.name for p in right_graphs},
            key=self.graph_sort_key,
        )

        if not common_graphs:
            st.warning("No common graphs found for the selected runs.")
            st.stop()

        graph_col, pdf_col = st.columns(2)
        with graph_col:
            graph_name = st.selectbox("Graph (common subset)", common_graphs)

        left_graph_path = subset_path / left_run_name / "graphs" / graph_name
        right_graph_path = subset_path / right_run_name / "graphs" / graph_name

        left_pdfs = self.list_pdfs(left_graph_path)
        right_pdfs = self.list_pdfs(right_graph_path)

        left_pdf_by_counter = {}
        for name in (p.name for p in left_pdfs):
            counter = self.pdf_counter(name)
            if counter is not None and counter not in left_pdf_by_counter:
                left_pdf_by_counter[counter] = name

        right_pdf_by_counter = {}
        for name in (p.name for p in right_pdfs):
            counter = self.pdf_counter(name)
            if counter is not None and counter not in right_pdf_by_counter:
                right_pdf_by_counter[counter] = name

        common_counters = sorted(set(left_pdf_by_counter) & set(right_pdf_by_counter))

        if not common_counters:
            st.warning("No common PDF files found in the selected graph.")
            st.stop()

        with pdf_col:
            pdf_counter_value = st.number_input(
                "PDF index (common subset)",
                min_value=min(common_counters),
                max_value=max(common_counters),
                value=common_counters[0],
                step=1,
            )

        if pdf_counter_value not in left_pdf_by_counter or pdf_counter_value not in right_pdf_by_counter:
            st.warning("Selected PDF index is not available in both runs.")
            st.stop()

        left_pdf_name = left_pdf_by_counter[pdf_counter_value]
        right_pdf_name = right_pdf_by_counter[pdf_counter_value]

        return (
            subset_path,
            left_run_name,
            right_run_name,
            left_pdf_name,
            right_pdf_name,
            graph_name,
        )

    def create_graphs(
        self,
        subset_path: Path,
        left_run_name: str,
        right_run_name: str,
        left_pdf_name: str,
        right_pdf_name: str,
        graph_name: str,
    ) -> None:
        left_graph_path = subset_path / left_run_name / "graphs" / graph_name
        right_graph_path = subset_path / right_run_name / "graphs" / graph_name

        left_pdf = left_graph_path / left_pdf_name
        right_pdf = right_graph_path / right_pdf_name

        st.caption(f"Rendering from: `{self.base_dir}`")

        left_col, right_col = st.columns(2)
        with left_col:
            st.subheader(f"{left_run_name} / {graph_name} / {left_pdf_name}")
            self.embed_pdf(left_pdf)
        with right_col:
            st.subheader(f"{right_run_name} / {graph_name} / {right_pdf_name}")
            self.embed_pdf(right_pdf)

    def create_metrics(
        self,
        subset_path: Path,
        left_run_name: str,
        right_run_name: str,
        graph_name: str,
    ) -> None:
        graph_index = self.graph_index(graph_name)
        if graph_index is None:
            st.warning("Unable to extract graph index for metric comparison.")
            return
        left_metrics_path = subset_path / left_run_name / "sample_metrics.csv"
        right_metrics_path = subset_path / right_run_name / "sample_metrics.csv"
        left_metrics = self.load_sample_metrics(left_metrics_path, graph_index)
        right_metrics = self.load_sample_metrics(right_metrics_path, graph_index)
        if left_metrics is None or right_metrics is None:
            st.warning("Metrics unavailable for the selected graph.")
            return
        metric_cols = st.columns(3)
        for col, key, label in zip(
            metric_cols,
            ["auroc", "auprc", "f1"],
            ["AUROC", "AUPRC", "F1"],
        ):
            left_value = left_metrics[key]
            right_value = right_metrics[key]
            delta = left_value - right_value
            col.metric(
                label,
                value=f"{left_value:.4f}",
                delta=f"{delta:+.4f}",
            )

    def run(self) -> None:
        (
            subset_path,
            left_run_name,
            right_run_name,
            left_pdf_name,
            right_pdf_name,
            graph_name,
        ) = self.create_selections()
        self.create_graphs(
            subset_path,
            left_run_name,
            right_run_name,
            left_pdf_name,
            right_pdf_name,
            graph_name,
        )
        self.create_metrics(
            subset_path,
            left_run_name,
            right_run_name,
            graph_name,
        )


if __name__ == "__main__":
    if not EXPLANATIONS_DIR.exists():
        st.error(f"Missing explanations directory: `{EXPLANATIONS_DIR}`")
        st.stop()

    st.set_page_config(page_title="Explanation Graph Viewer", layout="wide")
    st.title("Explanation Graph Viewer")
    viewer = ExplanationGraphViewer(EXPLANATIONS_DIR)
    viewer.run()
