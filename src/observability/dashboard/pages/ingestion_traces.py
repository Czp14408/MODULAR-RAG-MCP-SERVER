"""Dashboard Ingestion Trace 页面。"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render(context: Dict[str, Any]) -> None:
    st.title("Ingestion Traces")
    trace_service = context["trace_service"]
    traces = trace_service.list_traces(trace_type="ingestion")
    if not traces:
        st.info("暂无 ingestion trace。")
        return

    selected = st.selectbox(
        "Trace",
        options=range(len(traces)),
        format_func=lambda idx: f"{traces[idx].get('started_at')} | {traces[idx].get('trace_id')}",
    )
    trace = traces[selected]
    st.json(trace)
    st.bar_chart(
        {
            stage["stage"]: float(stage.get("elapsed_ms", 0.0))
            for stage in trace.get("stages", [])
            if stage.get("stage") in {"load", "split", "transform", "embed", "upsert"}
        }
    )
