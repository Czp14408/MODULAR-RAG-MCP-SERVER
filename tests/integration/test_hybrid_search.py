"""D5: HybridSearch 集成测试。"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.fusion import Fusion
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.loader.file_integrity import SQLiteIntegrityChecker


def _settings(tmp_path: Path) -> dict:
    return {
        "llm": {"provider": "placeholder"},
        "embedding": {"provider": "hash", "dimension": 8},
        "vector_store": {
            "provider": "chroma",
            "persist_directory": str(tmp_path / "data" / "db" / "chroma"),
        },
        "splitter": {"provider": "recursive", "chunk_size": 120, "chunk_overlap": 20},
        "ingestion": {
            "chunk_refiner": {"use_llm": False},
            "metadata_enricher": {"use_llm": False},
            "image_captioner": {"enabled": False},
            "batch_processor": {"batch_size": 2},
        },
        "rerank": {"enabled": False},
    }


@pytest.mark.integration
def test_hybrid_search_returns_top_k_and_supports_filters(tmp_path: Path) -> None:
    pdf_path = PROJECT_ROOT / "tests" / "data" / "test_chunking_multimodal.pdf"
    settings = _settings(tmp_path)
    bm25 = BM25Indexer(str(tmp_path / "data" / "db" / "bm25"))
    pipeline = IngestionPipeline(settings=settings)
    pipeline = IngestionPipeline(
        settings=settings,
        integrity_checker=SQLiteIntegrityChecker(str(tmp_path / "data" / "db" / "ingestion_history.db")),
        bm25_indexer=bm25,
    )
    pipeline.run(str(pdf_path), collection="demo", force=True)

    search = HybridSearch(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=DenseRetriever(settings),
        sparse_retriever=SparseRetriever(
            settings,
            bm25_indexer=bm25,
        ),
        fusion=Fusion(),
    )

    results = search.search("分布式 分片", top_k=3, filters={"collection": "demo"})
    print(f"[D5] filtered_results={[item.to_dict() for item in results]}")

    assert results
    assert len(results) <= 3
    assert all(item.metadata.get("collection") == "demo" for item in results)


@pytest.mark.integration
def test_hybrid_search_degrades_to_single_path_when_dense_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = PROJECT_ROOT / "tests" / "data" / "test_chunking_text.pdf"
    settings = _settings(tmp_path)
    bm25 = BM25Indexer(str(tmp_path / "data" / "db" / "bm25"))
    pipeline = IngestionPipeline(
        settings=settings,
        integrity_checker=SQLiteIntegrityChecker(str(tmp_path / "data" / "db" / "ingestion_history.db")),
        bm25_indexer=bm25,
    )
    pipeline.run(str(pdf_path), collection="demo", force=True)

    dense = DenseRetriever(settings)
    sparse = SparseRetriever(settings, bm25_indexer=bm25)
    search = HybridSearch(settings, QueryProcessor(), dense, sparse, Fusion())

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("dense unavailable")

    monkeypatch.setattr(dense, "retrieve", _boom)
    results = search.search("大语言模型", top_k=2)
    print(f"[D5] degraded_results={[item.to_dict() for item in results]}")
    print(f"[D5] debug={search.last_debug}")

    assert results
    assert search.last_debug["dense_error"] == "dense unavailable"
