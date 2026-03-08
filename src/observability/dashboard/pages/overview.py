"""Dashboard 系统总览页。"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render(context: Dict[str, Any]) -> None:
    st.title("System Overview")
    st.caption("当前组件配置与本地数据规模总览。")

    config = context["config_service"].summarize()
    stats = context["data_service"].get_collection_stats()

    cols = st.columns(4)
    cols[0].metric("Documents", stats["document_count"])
    cols[1].metric("Chunks", stats["chunk_count"])
    cols[2].metric("Images", stats["image_count"])
    cols[3].metric("Collections", len(stats["collections"]))

    st.subheader("Component Config")
    for name, payload in config.items():
        with st.expander(name, expanded=(name in {"embedding", "vector_store", "splitter"})):
            st.json(payload)
