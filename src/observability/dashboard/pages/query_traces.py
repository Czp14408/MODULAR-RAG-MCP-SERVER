"""Dashboard Query Trace 页面。"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render(context: Dict[str, Any]) -> None:
    st.title("Query Traces")
    trace_service = context["trace_service"]
    keyword = st.text_input("Search Query Keyword", value="")
    traces = trace_service.search_query_traces(keyword=keyword)
    if not traces:
        st.info("暂无 query trace。")
        return

    selected = st.selectbox(
        "Trace",
        options=range(len(traces)),
        format_func=lambda idx: f"{traces[idx].get('started_at')} | {traces[idx].get('trace_id')}",
    )
    trace = traces[selected]
    st.json(trace)
    stage_map = {stage["stage"]: float(stage.get("elapsed_ms", 0.0)) for stage in trace.get("stages", [])}
    st.bar_chart(
        {
            "query_processing": stage_map.get("query_processing", 0.0),
            "dense_retrieval": stage_map.get("dense_retrieval", 0.0),
            "sparse_retrieval": stage_map.get("sparse_retrieval", 0.0),
            "fusion": stage_map.get("fusion", 0.0),
            "rerank": stage_map.get("rerank", 0.0),
        }
    )
