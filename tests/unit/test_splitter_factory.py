"""B3: SplitterFactory 工厂路由测试。"""

from pathlib import Path
import sys
from typing import Any, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.recursive_splitter import RecursiveSplitter
from src.libs.splitter.splitter_factory import SplitterFactory, SplitterFactoryError


class FakeSplitter(BaseSplitter):
    # 测试假实现：返回可预测结果用于验证工厂分流。
    def split_text(self, text: str, trace: Optional[Any] = None) -> List[str]:
        return [f"fake::{text}"] if text else []


def test_factory_routes_to_registered_fake_provider() -> None:
    provider_name = "fake-splitter"
    SplitterFactory.register_provider(provider_name, FakeSplitter)

    splitter = SplitterFactory.create({"splitter": {"provider": provider_name}})

    assert isinstance(splitter, FakeSplitter)
    assert splitter.split_text("abc") == ["fake::abc"]


def test_factory_uses_builtin_recursive_provider() -> None:
    splitter = SplitterFactory.create(
        {"splitter": {"provider": "recursive", "chunk_size": 4, "chunk_overlap": 1}}
    )

    assert isinstance(splitter, RecursiveSplitter)
    chunks = splitter.split_text("abcdefghij")
    assert chunks


def test_factory_raises_on_unknown_provider() -> None:
    with pytest.raises(SplitterFactoryError, match="Unsupported splitter.provider"):
        SplitterFactory.create({"splitter": {"provider": "unknown"}})


def test_factory_raises_on_missing_provider() -> None:
    with pytest.raises(SplitterFactoryError, match="splitter.provider"):
        SplitterFactory.create({"splitter": {}})
