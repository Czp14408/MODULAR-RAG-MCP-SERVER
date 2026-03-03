"""C5: ChunkRefiner LLM 集成测试。"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types import Chunk
from src.ingestion.transform.chunk_refiner import ChunkRefiner


def _chunk(text: str) -> Chunk:
    return Chunk(
        id="c-int",
        text=text,
        metadata={"source_path": "tests/data/test_chunking_text.pdf"},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-int",
    )


@pytest.mark.integration
def test_chunk_refiner_real_llm_call_when_api_key_available() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set; skipping real LLM integration test")

    settings = {
        "llm": {
            "provider": "openai",
            "api_key": api_key,
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "timeout_seconds": 60,
        },
        "ingestion": {"chunk_refiner": {"use_llm": True}},
    }
    refiner = ChunkRefiner(settings=settings)
    chunk = _chunk("第 1 页\n\nRAG    系统  用于减少幻觉。\n---")
    out = refiner.transform([chunk])[0]
    print(f"[C5-INT] real llm output={out.text}")
    print(f"[C5-INT] metadata={out.metadata}")

    assert out.text.strip()
    assert out.metadata["refined_by"] in {"llm", "rule"}


@pytest.mark.integration
def test_chunk_refiner_invalid_provider_falls_back_to_rule() -> None:
    settings = {
        "llm": {"provider": "non-existent-provider"},
        "ingestion": {"chunk_refiner": {"use_llm": True}},
    }
    refiner = ChunkRefiner(settings=settings)
    out = refiner.transform([_chunk("第 2 页\n核心文本保留")])[0]
    print(f"[C5-INT] fallback output={out.text} metadata={out.metadata}")

    assert out.metadata["refined_by"] == "rule"
    assert "refine_fallback_reason" in out.metadata
    assert "核心文本保留" in out.text

