"""B8: Vision LLM 抽象与工厂路由测试。"""

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.llm.base_vision_llm import BaseVisionLLM, ChatResponse
from src.libs.llm.llm_factory import LLMFactory, LLMFactoryError


class FakeVisionLLM(BaseVisionLLM):
    # 测试桩：返回固定结果，验证工厂路由逻辑。
    def chat_with_image(self, text: str, image_path, trace=None) -> ChatResponse:  # noqa: ANN001
        return ChatResponse(content=f"fake-vision:{text}")


def test_factory_routes_to_registered_fake_vision_provider() -> None:
    provider_name = "fake_vision"
    LLMFactory.register_vision_provider(provider_name, FakeVisionLLM)

    vision_llm = LLMFactory.create_vision_llm({"vision_llm": {"provider": provider_name}})
    result = vision_llm.chat_with_image("hello", b"img-bytes")

    print(f"[B8] vision_provider={provider_name} result={result}")
    assert isinstance(vision_llm, FakeVisionLLM)
    assert result.content == "fake-vision:hello"


def test_factory_raises_on_missing_vision_provider() -> None:
    with pytest.raises(LLMFactoryError, match="vision_llm.provider"):
        LLMFactory.create_vision_llm({"vision_llm": {}})


def test_factory_raises_on_unknown_vision_provider() -> None:
    with pytest.raises(LLMFactoryError, match="Unsupported vision_llm.provider"):
        LLMFactory.create_vision_llm({"vision_llm": {"provider": "unknown"}})
