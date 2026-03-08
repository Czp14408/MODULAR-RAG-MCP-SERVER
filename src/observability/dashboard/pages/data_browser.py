"""Dashboard 数据浏览页。"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render(context: Dict[str, Any]) -> None:
    st.title("Data Browser")
    data_service = context["data_service"]
    stats = data_service.get_collection_stats()
    collections = ["all", *stats["collections"]]
    selected = st.selectbox("Collection", collections, index=0)
    collection = None if selected == "all" else selected

    documents = data_service.list_documents(collection=collection)
    st.caption(f"Documents: {len(documents)}")
    if not documents:
        st.info("暂无已摄入文档。")
        return

    for doc in documents:
        label = f"{doc['doc_id']} | {doc['source_path']}"
        with st.expander(label):
            st.write(
                {
                    "collection": doc["collection"],
                    "title": doc["title"],
                    "summary": doc["summary"],
                    "tags": doc["tags"],
                    "chunk_count": doc["chunk_count"],
                    "image_count": doc["image_count"],
                }
            )
            detail = data_service.get_document_detail(doc["doc_id"])
            st.write("chunks")
            for chunk in detail["chunks"]:
                st.code(str(chunk.get("text", ""))[:400])
            if detail["images"]:
                st.write("images")
                for image in detail["images"]:
                    st.write(image)
