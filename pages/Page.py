import streamlit as st


class AppPage:
    def __init__(self, base_dir: str) -> None:
        print(st.session_state)
        self.base_dir = base_dir
        self.subset = st.session_state.get("subset", None)
        self.left_run_name = st.session_state.get("left_run_name", None)
        self.right_run_name = st.session_state.get("right_run_name", None)
