import streamlit as st

pg = st.navigation(
    [
        st.Page("viewer/Analysis.py", title="Analysis", icon="📊"),
        st.Page("viewer/GraphViewer.py", title="Graph Viewer", icon="🔍"),
        st.Page("viewer/Config.py", title="Config", icon="⚙️"),
    ]
)
pg.run()
