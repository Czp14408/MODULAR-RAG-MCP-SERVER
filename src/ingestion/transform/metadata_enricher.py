"""MetadataEnricher：规则增强 + 可选 LLM 增强。"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, List, Optional

from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_llm import BaseLLM, Message
from src.libs.llm.llm_factory import LLMFactory


class MetadataEnricher(BaseTransform):
    """为 Chunk 补充 title/summary/tags 元数据。"""

    def __init__(self, settings: Any, llm: Optional[BaseLLM] = None) -> None:
        self.settings = settings
        self.use_llm = self._read_use_llm(settings)
        self._llm: Optional[BaseLLM] = llm
        self._llm_init_error: Optional[str] = None
        if self.use_llm and self._llm is None:
            try:
                self._llm = LLMFactory.create(settings)
            except Exception as exc:  # noqa: BLE001
                self._llm_init_error = str(exc)
                self._llm = None

    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        return self.enrich(chunks, trace=trace)

    def enrich(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        enriched: List[Chunk] = []
        llm_success = 0
        llm_fallback = 0

        for chunk in chunks:
            base = self._rule_enrich(chunk.text)
            meta = dict(chunk.metadata)
            meta.update(base)
            meta["metadata_enriched_by"] = "rule"

            if self.use_llm:
                llm_meta = self._llm_enrich(chunk.text)
                if llm_meta is not None:
                    meta.update(llm_meta)
                    meta["metadata_enriched_by"] = "llm"
                    llm_success += 1
                else:
                    llm_fallback += 1
                    if self._llm_init_error:
                        meta["metadata_enrich_fallback_reason"] = self._llm_init_error
            enriched.append(replace(chunk, metadata=meta))

        if trace is not None:
            trace.record_stage(
                "metadata_enricher",
                elapsed_ms=0.0,
                chunk_count=len(chunks),
                llm_enabled=self.use_llm,
                llm_success=llm_success,
                llm_fallback=llm_fallback,
            )
        return enriched

    def _rule_enrich(self, text: str) -> dict:
        clean = (text or "").strip()
        summary = clean[:120] if clean else "N/A"
        title = clean.splitlines()[0][:40] if clean else "N/A"
        tags = self._extract_tags(clean)
        return {"title": title, "summary": summary, "tags": tags}

    def _llm_enrich(self, text: str) -> Optional[dict]:
        if not self.use_llm or self._llm is None:
            return None
        prompt = (
            "请从以下文本中生成 JSON，字段仅包含 title, summary, tags。"
            "其中 tags 为字符串数组。\n\n"
            f"文本：{text}"
        )
        try:
            result = self._llm.chat(
                [
                    Message(role="system", content="你是一个信息抽取助手。"),
                    Message(role="user", content=prompt),
                ]
            )
            data = json.loads(result)
            if not isinstance(data, dict):
                return None
            title = str(data.get("title", "")).strip()
            summary = str(data.get("summary", "")).strip()
            tags_raw = data.get("tags", [])
            if not isinstance(tags_raw, list):
                tags_raw = []
            tags = [str(x).strip() for x in tags_raw if str(x).strip()]
            if not title or not summary:
                return None
            return {"title": title, "summary": summary, "tags": tags}
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _extract_tags(text: str) -> List[str]:
        # 参数选择说明：
        # 规则标签使用关键词命中，作为 LLM 不可用时的稳定兜底策略。
        candidates = [
            ("rag", "RAG"),
            ("embedding", "Embedding"),
            ("向量", "向量检索"),
            ("数据库", "数据库"),
            ("分布式", "分布式系统"),
            ("llm", "LLM"),
        ]
        lowered = text.lower()
        tags: List[str] = []
        for needle, tag in candidates:
            if needle in lowered or needle in text:
                tags.append(tag)
        return tags or ["General"]

    @staticmethod
    def _read_use_llm(settings: Any) -> bool:
        if isinstance(settings, dict):
            ingestion = settings.get("ingestion", {})
            if isinstance(ingestion, dict):
                enricher = ingestion.get("metadata_enricher", {})
                if isinstance(enricher, dict):
                    return bool(enricher.get("use_llm", False))
        if hasattr(settings, "ingestion"):
            ing = settings.ingestion
            if hasattr(ing, "metadata_enricher") and hasattr(ing.metadata_enricher, "use_llm"):
                return bool(ing.metadata_enricher.use_llm)
        return False
