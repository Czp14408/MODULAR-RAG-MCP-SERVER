"""B7.8: Cross-Encoder Reranker 测试（mock scorer）。"""

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.reranker.base_reranker import RerankerFallbackSignal
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from src.libs.reranker.reranker_factory import RerankerFactory


def test_factory_creates_cross_encoder_reranker() -> None:
    reranker = RerankerFactory.create({"rerank": {"provider": "cross_encoder"}})
    assert isinstance(reranker, CrossEncoderReranker)


def test_cross_encoder_reranker_scores_top_m_with_mock_scorer() -> None:
    def fake_scorer(query: str, text: str) -> float:
        # 通过文本长度构造确定性分数。
        return float(len(text))

    reranker = CrossEncoderReranker({"rerank": {"top_m": 2, "scorer": fake_scorer}})

    candidates = [
        {"id": "c1", "text": "aaaa"},
        {"id": "c2", "text": "bb"},
        {"id": "c3", "text": "cccccc"},
    ]

    ranked = reranker.rerank("q", candidates)

    # 仅前 2 个参与打分排序：c1(4) 在 c2(2) 前，c3 保持尾部原顺序。
    assert [item["id"] for item in ranked] == ["c1", "c2", "c3"]


def test_cross_encoder_reranker_timeout_returns_fallback_signal() -> None:
    def timeout_scorer(query: str, text: str) -> float:
        raise TimeoutError("scoring timeout")

    reranker = CrossEncoderReranker({"rerank": {"top_m": 2, "scorer": timeout_scorer}})

    with pytest.raises(RerankerFallbackSignal, match="TimeoutError"):
        reranker.rerank("q", [{"id": "c1", "text": "x"}, {"id": "c2", "text": "y"}])
