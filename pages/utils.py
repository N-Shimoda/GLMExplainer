import streamlit as st


def selectbox_with_state(
    label: str,
    options: list[str],
    state_key: str,
    default_index: int = 0,
    *,
    key: str | None = None,
    container: object | None = None,
) -> str:
    """Create a selectbox synced to a canonical session_state value."""
    if not options:
        return ""
    widget_key = key or f"{state_key}_widget"
    if st.session_state.get(state_key) not in options:
        st.session_state[state_key] = options[min(default_index, len(options) - 1)]
    desired = st.session_state[state_key]
    if st.session_state.get(widget_key) not in options or st.session_state.get(widget_key) != desired:
        st.session_state[widget_key] = desired

    def _sync_state() -> None:
        st.session_state[state_key] = st.session_state[widget_key]

    target = container if container is not None else st
    return target.selectbox(label, options, key=widget_key, on_change=_sync_state)
