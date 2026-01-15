import streamlit as st

pg = st.navigation(
    [
        st.Page("viewer/Analysis.py", title="Analysis", icon="📊"),
        st.Page("viewer/GraphViewer.py", title="Graph Viewer", icon="🔍"),
    ]
)
pg.run()
