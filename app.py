import streamlit as st

pg = st.navigation(
    [
        st.Page("pages/table_viewer.py", title="Analysis", icon="📊"),
        st.Page("pages/graph_viewer.py", title="Graph Viewer", icon="🔍"),
    ]
)
pg.run()
