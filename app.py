import streamlit as st

pg = st.navigation(
    [
        st.Page("pages/Analysis.py", title="Analysis", icon="📊"),
        st.Page("pages/GraphViewer.py", title="Graph Viewer", icon="🔍"),
    ]
)
pg.run()
