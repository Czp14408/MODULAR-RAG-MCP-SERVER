"""IngestionPipeline：串联 C2-C13 各组件的最小 MVP 编排。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, ChunkRecord
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.loader.pdf_loader import PdfLoader


class IngestionPipelineError(RuntimeError):
    """Pipeline 某阶段失败时抛出的异常。"""


class IngestionPipeline:
    """离线摄取 MVP：integrity -> load -> split -> transform -> embed -> upsert。"""

    def __init__(
        self,
        settings: Any,
        integrity_checker: Optional[SQLiteIntegrityChecker] = None,
        loader: Optional[PdfLoader] = None,
        chunker: Optional[DocumentChunker] = None,
        refiner: Optional[ChunkRefiner] = None,
        metadata_enricher: Optional[MetadataEnricher] = None,
        image_captioner: Optional[ImageCaptioner] = None,
        dense_encoder: Optional[DenseEncoder] = None,
        sparse_encoder: Optional[SparseEncoder] = None,
        batch_processor: Optional[BatchProcessor] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        vector_upserter: Optional[VectorUpserter] = None,
        image_storage: Optional[ImageStorage] = None,
    ) -> None:
        self.settings = settings
        self.integrity_checker = integrity_checker or SQLiteIntegrityChecker()
        self.loader = loader or PdfLoader()
        self.chunker = chunker or DocumentChunker(settings)
        self.refiner = refiner or ChunkRefiner(settings)
        self.metadata_enricher = metadata_enricher or MetadataEnricher(settings)
        self.image_captioner = image_captioner or ImageCaptioner(settings)
        self.dense_encoder = dense_encoder or DenseEncoder(settings)
        self.sparse_encoder = sparse_encoder or SparseEncoder()
        self.batch_processor = batch_processor or BatchProcessor(
            settings=settings,
            dense_encoder=self.dense_encoder,
            sparse_encoder=self.sparse_encoder,
        )
        self.bm25_indexer = bm25_indexer or BM25Indexer()
        self.vector_upserter = vector_upserter or VectorUpserter(settings)
        self.image_storage = image_storage or ImageStorage()

    def run(
        self,
        path: str,
        collection: str = "default",
        force: bool = False,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        pdf_path = Path(path)
        if not pdf_path.exists():
            raise IngestionPipelineError(f"input file not found: {pdf_path}")

        trace = TraceContext(trace_type="ingestion")
        file_hash = self.integrity_checker.compute_sha256(str(pdf_path))
        if not force and self.integrity_checker.should_skip(file_hash):
            self._emit_progress(on_progress, "integrity", 1, 1)
            return {
                "status": "skipped",
                "reason": "already_ingested",
                "file_hash": file_hash,
                "path": str(pdf_path),
            }

        try:
            document = self.loader.load(str(pdf_path))
            self._emit_progress(on_progress, "load", 1, 1)

            chunks = self.chunker.split_document(document)
            self._emit_progress(on_progress, "split", len(chunks), len(chunks) or 1)

            transformed = self._transform_chunks(chunks, trace=trace)
            self._emit_progress(
                on_progress, "transform", len(transformed), len(transformed) or 1
            )

            records = self.batch_processor.process(transformed, trace=trace)
            self._emit_progress(on_progress, "embed", len(records), len(records) or 1)

            sparse_only = [
                ChunkRecord(
                    id=record.id,
                    text=record.text,
                    metadata=dict(record.metadata),
                    sparse_vector=dict(record.sparse_vector or {}),
                )
                for record in records
            ]
            self.bm25_indexer.update(sparse_only)

            upserted_records = self.vector_upserter.upsert(records, trace=trace)
            stored_images = self._store_images(document, collection=collection)
            self._emit_progress(
                on_progress, "upsert", len(upserted_records), len(upserted_records) or 1
            )

            self.integrity_checker.mark_success(
                file_hash=file_hash,
                file_path=str(pdf_path),
                file_size=pdf_path.stat().st_size,
                chunk_count=len(upserted_records),
            )

            return {
                "status": "success",
                "file_hash": file_hash,
                "document_id": document.id,
                "chunk_count": len(upserted_records),
                "stored_images": stored_images,
                "trace": trace.to_dict(),
            }
        except Exception as exc:  # noqa: BLE001
            self.integrity_checker.mark_failed(
                file_hash=file_hash,
                error_msg=str(exc),
                file_path=str(pdf_path),
                file_size=pdf_path.stat().st_size if pdf_path.exists() else None,
            )
            raise IngestionPipelineError(f"pipeline failed at path={pdf_path}: {exc}") from exc

    def _transform_chunks(self, chunks: List[Chunk], trace: TraceContext) -> List[Chunk]:
        refined = self.refiner.transform(chunks, trace=trace)
        enriched = self.metadata_enricher.enrich(refined, trace=trace)
        captioned = self.image_captioner.caption(enriched, trace=trace)
        return captioned

    def _store_images(self, document: Any, collection: str) -> int:
        metadata = getattr(document, "metadata", {})
        images = metadata.get("images", []) if isinstance(metadata, dict) else []
        if not isinstance(images, list):
            return 0

        stored = 0
        for image in images:
            if not isinstance(image, dict):
                continue
            image_path = Path(str(image.get("path", "")))
            if not image_path.exists():
                continue
            self.image_storage.save_image(
                image_id=str(image.get("id", "")),
                image_bytes=image_path.read_bytes(),
                collection=collection,
                doc_hash=str(getattr(document, "id", "")),
                page_num=int(image.get("page", 0)) if image.get("page") is not None else None,
                extension=image_path.suffix or ".png",
            )
            stored += 1
        return stored

    @staticmethod
    def _emit_progress(
        callback: Optional[Callable[[str, int, int], None]],
        stage_name: str,
        current: int,
        total: int,
    ) -> None:
        if callback is not None:
            callback(stage_name, current, total)
