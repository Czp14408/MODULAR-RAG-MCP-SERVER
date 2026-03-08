"""B5: RerankerFactory 测试。"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.reranker.base_reranker import BaseReranker, NoneReranker
from src.libs.reranker.reranker_factory import RerankerFactory, RerankerFactoryError


class FakeReranker(BaseReranker):
    # 测试桩：按 id 倒序返回，便于验证分流。
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        return sorted(candidates, key=lambda x: x.get("id", ""), reverse=True)


def test_none_reranker_keeps_order() -> None:
    reranker = RerankerFactory.create({"rerank": {"provider": "none"}})
    assert isinstance(reranker, NoneReranker)

    candidates = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert reranker.rerank("q", candidates) == candidates


def test_factory_defaults_to_none_when_disabled() -> None:
    reranker = RerankerFactory.create({"rerank": {"enabled": False}})
    assert isinstance(reranker, NoneReranker)


def test_factory_routes_to_registered_fake_provider() -> None:
    provider_name = "fake-reranker"
    RerankerFactory.register_provider(provider_name, FakeReranker)

    reranker = RerankerFactory.create({"rerank": {"provider": provider_name}})
    result = reranker.rerank("q", [{"id": "a"}, {"id": "c"}, {"id": "b"}])

    assert isinstance(reranker, FakeReranker)
    assert [item["id"] for item in result] == ["c", "b", "a"]


def test_factory_raises_on_unknown_provider() -> None:
    with pytest.raises(RerankerFactoryError, match="Unsupported rerank.provider"):
        RerankerFactory.create({"rerank": {"provider": "unknown"}})


def test_factory_raises_when_enabled_without_provider() -> None:
    with pytest.raises(RerankerFactoryError, match="rerank.provider"):
        RerankerFactory.create({"rerank": {"enabled": True}})
