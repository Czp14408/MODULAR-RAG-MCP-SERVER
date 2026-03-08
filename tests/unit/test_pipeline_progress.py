"""F5: Pipeline 进度回调测试。"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types import Chunk, ChunkRecord, Document
from src.ingestion.pipeline import IngestionPipeline
from src.libs.loader.file_integrity import SQLiteIntegrityChecker


class _FakeLoader:
    """返回固定文档，避免单测依赖真实 PDF 解析。"""

    def load(self, path: str) -> Document:
        return Document(id="doc-1", text="alpha beta gamma", metadata={"source_path": path})


class _FakeChunker:
    """返回固定 chunk，便于验证进度回调顺序。"""

    def split_document(self, document: Document) -> list[Chunk]:
        return [
            Chunk(
                id="chunk-1",
                text=document.text,
                metadata={"source_path": document.metadata["source_path"], "chunk_index": 0},
                start_offset=0,
                end_offset=len(document.text),
                source_ref=document.id,
            )
        ]


class _PassthroughTransform:
    def transform(self, chunks: list[Chunk], trace=None) -> list[Chunk]:  # noqa: ANN001
        return chunks


class _PassthroughMetadataEnricher:
    def enrich(self, chunks: list[Chunk], trace=None) -> list[Chunk]:  # noqa: ANN001
        return chunks


class _PassthroughImageCaptioner:
    def caption(self, chunks: list[Chunk], trace=None) -> list[Chunk]:  # noqa: ANN001
        return chunks


class _FakeBatchProcessor:
    def process(self, chunks: list[Chunk], trace=None) -> list[ChunkRecord]:  # noqa: ANN001
        chunk = chunks[0]
        return [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                dense_vector=[0.1, 0.2],
                sparse_vector={"alpha": 1.0},
            )
        ]


class _FakeBM25:
    def update(self, records: list[ChunkRecord]) -> dict:  # noqa: ARG002
        return {}


class _FakeUpserter:
    def upsert(self, records: list[ChunkRecord], trace=None) -> list[ChunkRecord]:  # noqa: ANN001
        return records


class _FakeImageStorage:
    def save_image(self, **kwargs) -> None:  # noqa: ANN003
        return None


class _FakeTraceCollector:
    def __init__(self) -> None:
        self.collected = []

    def collect(self, trace) -> None:  # noqa: ANN001
        self.collected.append(trace.to_dict())


class _FakeDenseEncoder:
    def encode(self, chunks: list[Chunk], trace=None) -> list[ChunkRecord]:  # noqa: ANN001
        return []


class _FakeSparseEncoder:
    def encode(self, chunks: list[Chunk], trace=None) -> list[ChunkRecord]:  # noqa: ANN001
        return []


def test_pipeline_on_progress_reports_each_stage(tmp_path: Path) -> None:
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    progress_events = []
    trace_collector = _FakeTraceCollector()

    pipeline = IngestionPipeline(
        settings={},
        integrity_checker=SQLiteIntegrityChecker(str(tmp_path / "ingestion_history.db")),
        loader=_FakeLoader(),
        chunker=_FakeChunker(),
        refiner=_PassthroughTransform(),
        metadata_enricher=_PassthroughMetadataEnricher(),
        image_captioner=_PassthroughImageCaptioner(),
        dense_encoder=_FakeDenseEncoder(),
        sparse_encoder=_FakeSparseEncoder(),
        batch_processor=_FakeBatchProcessor(),
        bm25_indexer=_FakeBM25(),
        vector_upserter=_FakeUpserter(),
        image_storage=_FakeImageStorage(),
        trace_collector=trace_collector,
    )

    result = pipeline.run(
        path=str(pdf_path),
        collection="demo",
        force=True,
        on_progress=lambda stage, current, total: progress_events.append((stage, current, total)),
    )
    print(f"[F5] progress_events={progress_events}")
    print(f"[F5] result={result}")

    assert [item[0] for item in progress_events] == ["load", "split", "transform", "embed", "upsert"]
    assert trace_collector.collected
    assert result["status"] == "success"
