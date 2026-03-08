"""C14: IngestionPipeline 集成测试。"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pipeline import IngestionPipeline, IngestionPipelineError
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.core.trace import TraceCollector
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.loader.pdf_loader import PdfLoader


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
    }


@pytest.mark.integration
def test_pipeline_runs_end_to_end_and_persists_outputs(tmp_path: Path) -> None:
    pdf_path = PROJECT_ROOT / "tests" / "data" / "test_chunking_multimodal.pdf"
    progress_events = []

    pipeline = IngestionPipeline(
        settings=_settings(tmp_path),
        integrity_checker=SQLiteIntegrityChecker(str(tmp_path / "data" / "db" / "ingestion_history.db")),
        loader=PdfLoader(images_root=str(tmp_path / "loader_images")),
        bm25_indexer=BM25Indexer(str(tmp_path / "data" / "db" / "bm25")),
        vector_upserter=VectorUpserter(_settings(tmp_path)),
        image_storage=ImageStorage(
            images_root=str(tmp_path / "data" / "images"),
            db_path=str(tmp_path / "data" / "db" / "image_index.db"),
        ),
        trace_collector=TraceCollector(str(tmp_path / "logs" / "traces.jsonl")),
    )

    result = pipeline.run(
        path=str(pdf_path),
        collection="tech-docs",
        force=True,
        on_progress=lambda stage, current, total: progress_events.append((stage, current, total)),
    )
    trace = result["trace"]

    chroma_store = tmp_path / "data" / "db" / "chroma" / "store.json"
    bm25_store = tmp_path / "data" / "db" / "bm25" / "bm25_index.json"
    image_db = tmp_path / "data" / "db" / "image_index.db"
    trace_log = tmp_path / "logs" / "traces.jsonl"
    print(f"[C14] result={result}")
    print(f"[C14] progress_events={progress_events}")

    assert result["status"] == "success"
    assert result["chunk_count"] > 0
    assert result["stored_images"] > 0
    assert chroma_store.exists()
    assert bm25_store.exists()
    assert image_db.exists()
    assert [item[0] for item in progress_events] == ["load", "split", "transform", "embed", "upsert"]
    assert trace["trace_type"] == "ingestion"
    stage_names = [item["stage"] for item in trace["stages"]]
    for expected in ["load", "split", "transform", "embed", "upsert"]:
        assert expected in stage_names
    top_level_stages = [item for item in trace["stages"] if item["stage"] in {"load", "split", "transform", "embed", "upsert"}]
    assert all("method" in item for item in top_level_stages)
    assert trace_log.exists()

    with sqlite3.connect(str(image_db)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM image_index").fetchone()[0]
    assert count >= 1


@pytest.mark.integration
def test_pipeline_raises_clear_error_for_missing_file(tmp_path: Path) -> None:
    pipeline = IngestionPipeline(settings=_settings(tmp_path))

    with pytest.raises(IngestionPipelineError, match="input file not found"):
        pipeline.run(path=str(tmp_path / "missing.pdf"), collection="demo")
