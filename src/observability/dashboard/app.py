"""Streamlit Dashboard 入口。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from src.core.settings import load_settings
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.trace import TraceCollector
from src.ingestion.document_manager import DocumentManager
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.vector_store.chroma_store import ChromaStore
from src.observability.dashboard.pages import (
    data_browser,
    evaluation_panel,
    ingestion_manager,
    ingestion_traces,
    overview,
    query_traces,
)
from src.observability.dashboard.services.config_service import ConfigService
from src.observability.dashboard.services.data_service import DataService
from src.observability.dashboard.services.trace_service import TraceService


def build_context() -> Dict[str, Any]:
    settings_path = os.environ.get("DASHBOARD_SETTINGS_PATH", "config/settings.yaml")
    settings = load_settings(settings_path)
    data_root = Path(os.environ.get("DASHBOARD_DATA_ROOT", "data"))
    trace_path = os.environ.get("DASHBOARD_TRACE_LOG", "logs/traces.jsonl")

    chroma = ChromaStore(
        {
            "vector_store": {
                "provider": "chroma",
                "persist_directory": str(data_root / "db" / "chroma"),
            }
        }
    )
    bm25 = BM25Indexer(str(data_root / "db" / "bm25"))
    image_storage = ImageStorage(
        images_root=str(data_root / "images"),
        db_path=str(data_root / "db" / "image_index.db"),
    )
    integrity = SQLiteIntegrityChecker(str(data_root / "db" / "ingestion_history.db"))
    document_manager = DocumentManager(chroma, bm25, image_storage, integrity)
    hybrid_search = HybridSearch(settings)

    return {
        "settings": settings,
        "hybrid_search": hybrid_search,
        "config_service": ConfigService(settings),
        "data_service": DataService(document_manager),
        "trace_service": TraceService(trace_log_path=trace_path),
        "pipeline": IngestionPipeline(
            settings=settings,
            integrity_checker=integrity,
            bm25_indexer=bm25,
            image_storage=image_storage,
            trace_collector=TraceCollector(trace_path),
        ),
    }


def main() -> None:
    st.set_page_config(page_title="Modular RAG Dashboard", layout="wide")
    context = build_context()
    st.session_state["_dashboard_settings"] = context["settings"]
    st.session_state["_dashboard_hybrid_search"] = context["hybrid_search"]
    page_renderers = {
        "Overview": lambda: overview.render(context),
        "Data Browser": lambda: data_browser.render(context),
        "Ingestion Manager": lambda: ingestion_manager.render(context),
        "Ingestion Traces": lambda: ingestion_traces.render(context),
        "Query Traces": lambda: query_traces.render(context),
        "Evaluation Panel": evaluation_panel.render,
    }
    pages = [
        st.Page(page_renderers["Overview"], title="Overview", url_path="overview"),
        st.Page(page_renderers["Data Browser"], title="Data Browser", url_path="data-browser"),
        st.Page(page_renderers["Ingestion Manager"], title="Ingestion Manager", url_path="ingestion-manager"),
        st.Page(page_renderers["Ingestion Traces"], title="Ingestion Traces", url_path="ingestion-traces"),
        st.Page(page_renderers["Query Traces"], title="Query Traces", url_path="query-traces"),
        st.Page(page_renderers["Evaluation Panel"], title="Evaluation Panel", url_path="evaluation-panel"),
    ]
    test_page = os.environ.get("DASHBOARD_TEST_PAGE", "").strip()
    if test_page:
        renderer = page_renderers.get(test_page)
        if renderer is not None:
            renderer()
            return
        raise ValueError(f"unknown dashboard test page: {test_page}")

    navigation = st.navigation(pages)
    navigation.run()


if __name__ == "__main__":
    main()
