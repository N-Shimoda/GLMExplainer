from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

EXPLANATIONS_DIR = Path("explanations")


def list_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir()])


def list_pdfs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(path.glob("*.pdf"))


def embed_pdf(path: Path) -> None:
    pdf_bytes = path.read_bytes()
    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{b64}" width=100% height=420></iframe>',
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Explanation Graph Viewer", layout="wide")
st.title("Explanation Graph Viewer")

if not EXPLANATIONS_DIR.exists():
    st.error(f"Missing explanations directory: `{EXPLANATIONS_DIR}`")
    st.stop()

subset_dirs = list_dirs(EXPLANATIONS_DIR)
subset_names = [p.name for p in subset_dirs]
subset = st.selectbox("Subset", subset_names)

subset_path = EXPLANATIONS_DIR / subset
run_dirs = list_dirs(subset_path)
run_names = [p.name for p in run_dirs]

left_run, right_run = st.columns(2)
with left_run:
    left_run_name = st.selectbox("Left run", run_names, index=0)
with right_run:
    right_run_name = st.selectbox("Right run", run_names, index=min(1, len(run_names) - 1))

left_graphs = list_dirs(subset_path / left_run_name / "graphs")
right_graphs = list_dirs(subset_path / right_run_name / "graphs")
common_graphs = sorted({p.name for p in left_graphs} & {p.name for p in right_graphs})

if not common_graphs:
    st.warning("No common graphs found for the selected runs.")
    st.stop()

graph_name = st.selectbox("Graph (common subset)", common_graphs)

left_graph_path = subset_path / left_run_name / "graphs" / graph_name
right_graph_path = subset_path / right_run_name / "graphs" / graph_name

left_pdfs = list_pdfs(left_graph_path)
right_pdfs = list_pdfs(right_graph_path)
common_pdfs = sorted({p.name for p in left_pdfs} & {p.name for p in right_pdfs})

if not common_pdfs:
    st.warning("No common PDF files found in the selected graph.")
    st.stop()

pdf_name = st.selectbox("PDF (common subset)", common_pdfs)

left_pdf = left_graph_path / pdf_name
right_pdf = right_graph_path / pdf_name

st.caption(f"Rendering from: `{EXPLANATIONS_DIR}`")

left_col, right_col = st.columns(2)
with left_col:
    st.subheader(f"{left_run_name} / {graph_name} / {pdf_name}")
    embed_pdf(left_pdf)
with right_col:
    st.subheader(f"{right_run_name} / {graph_name} / {pdf_name}")
    embed_pdf(right_pdf)
