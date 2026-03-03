"""C5: ChunkRefiner 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.libs.llm.base_llm import BaseLLM, Message


class FakeLLM(BaseLLM):
    """可控 LLM 测试桩：返回固定文本。"""

    def __init__(self, settings, answer: str = "LLM 精炼结果", raise_error: bool = False) -> None:
        super().__init__(settings)
        self.answer = answer
        self.raise_error = raise_error
        self.last_messages: List[Message] = []

    def chat(self, messages: List[Message]) -> str:
        self.last_messages = messages
        if self.raise_error:
            raise RuntimeError("mock llm failure")
        return self.answer


def _chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(
        id=f"c{idx}",
        text=text,
        metadata={"source_path": "tests/data/test_chunking_text.pdf"},
        start_offset=0,
        end_offset=max(0, len(text)),
        source_ref="doc1",
    )


def _load_cases() -> list[dict]:
    fixture = PROJECT_ROOT / "tests" / "fixtures" / "noisy_chunks.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    return list(payload["cases"])


@pytest.mark.parametrize("case", _load_cases(), ids=[c["name"] for c in _load_cases()])
def test_rule_based_refine_handles_noisy_fixtures(case: dict) -> None:
    refiner = ChunkRefiner(settings={"ingestion": {"chunk_refiner": {"use_llm": False}}})
    cleaned = refiner._rule_based_refine(case["input"])
    print(f"[C5] case={case['name']} cleaned={cleaned}")

    for expected in case["expect_contains"]:
        assert expected in cleaned
    for forbidden in case["expect_not_contains"]:
        assert forbidden not in cleaned


def test_code_block_format_is_preserved() -> None:
    text = "说明\n```python\nx = 1\nif x:\n    print(x)\n```\n第 2 页"
    refiner = ChunkRefiner(settings={"ingestion": {"chunk_refiner": {"use_llm": False}}})
    cleaned = refiner._rule_based_refine(text)
    print(f"[C5] code cleaned={cleaned}")

    assert "```python" in cleaned
    assert "    print(x)" in cleaned
    assert "第 2 页" not in cleaned


def test_transform_rule_only_marks_metadata() -> None:
    chunk = _chunk("第 1 页\n核心内容")
    refiner = ChunkRefiner(settings={"ingestion": {"chunk_refiner": {"use_llm": False}}})
    output = refiner.transform([chunk])
    print(f"[C5] rule-only output={output[0].text} metadata={output[0].metadata}")

    assert len(output) == 1
    assert output[0].metadata["refined_by"] == "rule"
    assert "第 1 页" not in output[0].text


def test_transform_llm_mode_uses_mock_and_marks_llm() -> None:
    fake_llm = FakeLLM(settings={}, answer="这是 LLM 改写后的文本")
    refiner = ChunkRefiner(
        settings={"ingestion": {"chunk_refiner": {"use_llm": True}}},
        llm=fake_llm,
        prompt_path=str(PROJECT_ROOT / "config" / "prompts" / "chunk_refinement.txt"),
    )
    chunk = _chunk("原始文本")

    output = refiner.transform([chunk])
    print(f"[C5] llm output={output[0].text} metadata={output[0].metadata}")
    print(f"[C5] llm messages={fake_llm.last_messages}")

    assert output[0].text == "这是 LLM 改写后的文本"
    assert output[0].metadata["refined_by"] == "llm"
    assert len(fake_llm.last_messages) == 2


def test_transform_fallbacks_to_rule_when_llm_raises() -> None:
    fake_llm = FakeLLM(settings={}, raise_error=True)
    refiner = ChunkRefiner(
        settings={"ingestion": {"chunk_refiner": {"use_llm": True}}},
        llm=fake_llm,
    )
    chunk = _chunk("第 1 页\n保留文本")

    output = refiner.transform([chunk])
    print(f"[C5] fallback output={output[0].text} metadata={output[0].metadata}")

    assert output[0].metadata["refined_by"] == "rule"
    assert "refine_fallback_reason" in output[0].metadata
    assert "保留文本" in output[0].text


def test_transform_fallbacks_to_rule_when_llm_returns_empty() -> None:
    fake_llm = FakeLLM(settings={}, answer="   ")
    refiner = ChunkRefiner(
        settings={"ingestion": {"chunk_refiner": {"use_llm": True}}},
        llm=fake_llm,
    )
    chunk = _chunk("可读内容")

    output = refiner.transform([chunk])
    print(f"[C5] empty-llm fallback metadata={output[0].metadata}")

    assert output[0].metadata["refined_by"] == "rule"
    assert output[0].metadata["refine_fallback_reason"] == "empty_llm_output"


def test_settings_flag_controls_use_llm_behavior() -> None:
    chunk = _chunk("原始文本")
    llm_on = ChunkRefiner(
        settings={"ingestion": {"chunk_refiner": {"use_llm": True}}},
        llm=FakeLLM(settings={}, answer="LLM 文本"),
    )
    llm_off = ChunkRefiner(
        settings={"ingestion": {"chunk_refiner": {"use_llm": False}}},
        llm=FakeLLM(settings={}, answer="LLM 文本"),
    )

    out_on = llm_on.transform([chunk])[0]
    out_off = llm_off.transform([chunk])[0]
    print(f"[C5] llm_on={out_on.text} llm_off={out_off.text}")

    assert out_on.text == "LLM 文本"
    assert out_on.metadata["refined_by"] == "llm"
    assert out_off.metadata["refined_by"] == "rule"


def test_single_chunk_exception_does_not_block_others(monkeypatch: pytest.MonkeyPatch) -> None:
    refiner = ChunkRefiner(settings={"ingestion": {"chunk_refiner": {"use_llm": False}}})
    chunks = [_chunk("正常文本", 1), _chunk("触发异常文本", 2)]

    original = refiner._rule_based_refine

    def _faulty(text: str) -> str:
        if "触发异常" in text:
            raise RuntimeError("boom")
        return original(text)

    monkeypatch.setattr(refiner, "_rule_based_refine", _faulty)
    output = refiner.transform(chunks)
    print(f"[C5] exception isolation output={[c.metadata for c in output]}")

    assert len(output) == 2
    assert output[0].metadata["refined_by"] == "rule"
    assert output[1].metadata["refine_fallback_reason"].startswith("chunk_exception:")
    assert output[1].text == "触发异常文本"


def test_prompt_loads_default_when_file_missing() -> None:
    refiner = ChunkRefiner(
        settings={"ingestion": {"chunk_refiner": {"use_llm": False, "prompt_path": "missing.txt"}}}
    )
    assert "{text}" in refiner.prompt


def test_prompt_appends_text_placeholder_when_missing(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("仅有说明文字", encoding="utf-8")
    refiner = ChunkRefiner(
        settings={"ingestion": {"chunk_refiner": {"use_llm": False}}},
        prompt_path=str(prompt_file),
    )
    assert "{text}" in refiner.prompt


def test_trace_context_records_refiner_stage() -> None:
    trace = TraceContext(trace_type="ingestion")
    refiner = ChunkRefiner(settings={"ingestion": {"chunk_refiner": {"use_llm": False}}})
    output = refiner.transform([_chunk("第 1 页\n文本")], trace=trace)
    stage = trace.get_stage("chunk_refiner")
    print(f"[C5] trace stage={stage} output={output[0].text}")

    assert stage is not None
    assert stage["details"]["chunk_count"] == 1


def test_trace_context_records_llm_call_stage() -> None:
    trace = TraceContext(trace_type="ingestion")
    refiner = ChunkRefiner(
        settings={"llm": {"provider": "openai"}, "ingestion": {"chunk_refiner": {"use_llm": True}}},
        llm=FakeLLM(settings={}, answer="ok"),
    )
    _ = refiner.transform([_chunk("文本")], trace=trace)
    stage = trace.get_stage("chunk_refiner_llm_call")
    print(f"[C5] llm trace stage={stage}")

    assert stage is not None
    assert stage["details"]["provider"] == "openai"

