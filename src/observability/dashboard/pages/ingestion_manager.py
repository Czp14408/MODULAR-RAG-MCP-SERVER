"""Dashboard Ingestion 管理页。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st


def render(context: Dict[str, Any]) -> None:
    st.title("Ingestion Manager")
    st.caption("上传 PDF 触发摄取，并支持删除已摄入文档。")

    data_service = context["data_service"]
    pipeline = context["pipeline"]
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    collection = st.text_input("Collection", value="default")
    progress_bar = st.progress(0)

    if uploaded is not None and st.button("Run Ingestion"):
        temp_dir = Path(".streamlit_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded.name
        temp_path.write_bytes(uploaded.getvalue())
        events = []

        def _on_progress(stage: str, current: int, total: int) -> None:
            ratio = min(1.0, float(current) / float(total or 1))
            progress_bar.progress(ratio, text=f"{stage}: {current}/{total}")
            events.append((stage, current, total))

        result = pipeline.run(
            path=str(temp_path),
            collection=collection or "default",
            force=True,
            on_progress=_on_progress,
        )
        st.success(result["status"])
        st.json(result)
        st.write(events)

    st.subheader("Delete Document")
    documents = data_service.list_documents()
    if not documents:
        st.info("暂无可删除文档。")
        return

    options = {f"{doc['source_path']} | {doc['collection']}": doc for doc in documents}
    selected = st.selectbox("Document", list(options.keys()))
    if st.button("Delete Selected"):
        doc = options[selected]
        result = data_service.delete_document(doc["source_path"], collection=doc["collection"])
        st.warning("document deleted")
        st.json(result)
