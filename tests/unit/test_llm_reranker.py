"""B7.7: LLMReranker 测试（mock LLM）。"""

from pathlib import Path
import sys
from typing import Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.reranker.base_reranker import RerankerContractError, RerankerFallbackSignal
from src.libs.reranker.llm_reranker import LLMReranker
from src.libs.reranker.reranker_factory import RerankerFactory


class _FakeLLM:
    def __init__(self, output: Optional[str] = None, error: Optional[Exception] = None) -> None:
        self.output = output
        self.error = error

    def chat(self, _messages):  # noqa: ANN001
        if self.error is not None:
            raise self.error
        return self.output


def test_factory_creates_llm_reranker() -> None:
    reranker = RerankerFactory.create({"rerank": {"provider": "llm"}})
    assert isinstance(reranker, LLMReranker)


def test_llm_reranker_reads_prompt_and_ranks_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prompt_file = tmp_path / "rerank_prompt.txt"
    prompt_file.write_text("You are a reranker", encoding="utf-8")

    fake_llm = _FakeLLM(output='{"ranked_ids": ["c2", "c1"]}')
    monkeypatch.setattr("src.libs.reranker.llm_reranker.LLMFactory.create", lambda _settings: fake_llm)

    reranker = LLMReranker({"rerank": {"prompt_path": str(prompt_file)}})
    candidates = [
        {"id": "c1", "text": "short"},
        {"id": "c2", "text": "long"},
    ]

    ranked = reranker.rerank("query", candidates)
    assert [item["id"] for item in ranked] == ["c2", "c1"]


def test_llm_reranker_invalid_schema_raises_readable_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prompt_file = tmp_path / "rerank_prompt.txt"
    prompt_file.write_text("You are a reranker", encoding="utf-8")

    fake_llm = _FakeLLM(output='{"ranked_ids": [123]}')
    monkeypatch.setattr("src.libs.reranker.llm_reranker.LLMFactory.create", lambda _settings: fake_llm)

    reranker = LLMReranker({"rerank": {"prompt_path": str(prompt_file)}})

    with pytest.raises(RerankerContractError, match="ranked_ids"):
        reranker.rerank("query", [{"id": "c1", "text": "x"}])


def test_llm_reranker_llm_failure_returns_fallback_signal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prompt_file = tmp_path / "rerank_prompt.txt"
    prompt_file.write_text("You are a reranker", encoding="utf-8")

    fake_llm = _FakeLLM(error=RuntimeError("network down"))
    monkeypatch.setattr("src.libs.reranker.llm_reranker.LLMFactory.create", lambda _settings: fake_llm)

    reranker = LLMReranker({"rerank": {"prompt_path": str(prompt_file)}})

    with pytest.raises(RerankerFallbackSignal, match="LLMCallFailed"):
        reranker.rerank("query", [{"id": "c1", "text": "x"}])
