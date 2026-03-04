"""C6: MetadataEnricher 契约测试。"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types import Chunk
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.libs.llm.base_llm import BaseLLM, Message


class FakeLLM(BaseLLM):
    def __init__(self, settings, response: str, raise_error: bool = False) -> None:
        super().__init__(settings)
        self.response = response
        self.raise_error = raise_error
        self.last_messages: List[Message] = []

    def chat(self, messages: List[Message]) -> str:
        self.last_messages = messages
        if self.raise_error:
            raise RuntimeError("mock metadata llm failed")
        return self.response


def _chunk(text: str) -> Chunk:
    return Chunk(
        id="c-meta",
        text=text,
        metadata={"source_path": "tests/data/test_chunking_text.pdf"},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc1",
    )


def test_rule_mode_outputs_title_summary_tags() -> None:
    enricher = MetadataEnricher(settings={"ingestion": {"metadata_enricher": {"use_llm": False}}})
    out = enricher.enrich([_chunk("RAG 系统通过向量检索提升问答准确率。")])[0]
    print(f"[C6] rule metadata={out.metadata}")

    assert out.metadata["title"]
    assert out.metadata["summary"]
    assert isinstance(out.metadata["tags"], list) and out.metadata["tags"]
    assert out.metadata["metadata_enriched_by"] == "rule"


def test_llm_mode_uses_llm_result_when_available() -> None:
    fake = FakeLLM(
        settings={},
        response='{"title":"LLM标题","summary":"LLM摘要","tags":["RAG","LLM"]}',
    )
    enricher = MetadataEnricher(
        settings={"ingestion": {"metadata_enricher": {"use_llm": True}}},
        llm=fake,
    )
    out = enricher.enrich([_chunk("原始文本内容")])[0]
    print(f"[C6] llm metadata={out.metadata}")
    print(f"[C6] llm messages={fake.last_messages}")

    assert out.metadata["metadata_enriched_by"] == "llm"
    assert out.metadata["title"] == "LLM标题"
    assert out.metadata["summary"] == "LLM摘要"
    assert out.metadata["tags"] == ["RAG", "LLM"]


def test_llm_failure_fallbacks_to_rule_result() -> None:
    fake = FakeLLM(settings={}, response="{}", raise_error=True)
    enricher = MetadataEnricher(
        settings={"ingestion": {"metadata_enricher": {"use_llm": True}}},
        llm=fake,
    )
    out = enricher.enrich([_chunk("分布式系统与数据库分片。")])[0]
    print(f"[C6] fallback metadata={out.metadata}")

    assert out.metadata["metadata_enriched_by"] == "rule"
    assert out.metadata["title"]
    assert out.metadata["summary"]
    assert out.metadata["tags"]
