"""Ingestion 模块导出。"""

from src.ingestion.pipeline import IngestionPipeline, IngestionPipelineError

__all__ = ["IngestionPipeline", "IngestionPipelineError"]
