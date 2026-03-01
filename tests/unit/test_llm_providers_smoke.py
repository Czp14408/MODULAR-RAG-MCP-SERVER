"""B7.1: OpenAI-compatible LLM provider 冒烟测试（mock HTTP）。"""

from pathlib import Path
import io
import json
import sys
from typing import Any, Dict
from urllib.error import HTTPError

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.llm.base_llm import Message
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.openai_llm import OpenAILLMError


class _FakeHTTPResponse:
    """最小可用的 HTTP 响应桩，支持 with 上下文。"""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


@pytest.mark.parametrize(
    "provider, llm_config, expected_url_part",
    [
        (
            "openai",
            {
                "api_key": "sk-openai",
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
            },
            "/chat/completions",
        ),
        (
            "azure",
            {
                "api_key": "azure-key",
                "endpoint": "https://example.openai.azure.com",
                "deployment": "gpt-4o-mini",
                "api_version": "2024-10-21",
            },
            "/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-10-21",
        ),
        (
            "deepseek",
            {
                "api_key": "sk-deepseek",
                "model": "deepseek-chat",
                "base_url": "https://api.deepseek.com/v1",
            },
            "/chat/completions",
        ),
    ],
)
def test_openai_compatible_providers_chat_success(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    llm_config: Dict[str, Any],
    expected_url_part: str,
) -> None:
    captured = {"url": "", "body": {}}

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": f"reply-from-{provider}",
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("src.libs.llm.openai_llm.urlrequest.urlopen", fake_urlopen)

    llm = LLMFactory.create({"llm": {"provider": provider, **llm_config}})
    result = llm.chat([Message(role="user", content="hello")])

    assert result == f"reply-from-{provider}"
    assert expected_url_part in captured["url"]
    assert captured["body"]["messages"][0]["content"] == "hello"


def test_chat_validation_error_contains_provider_and_error_type() -> None:
    llm = LLMFactory.create(
        {
            "llm": {
                "provider": "openai",
                "api_key": "sk-openai",
            }
        }
    )

    with pytest.raises(OpenAILLMError, match=r"provider=openai.*error_type=ValidationError"):
        llm.chat([])


def test_http_error_is_wrapped_with_readable_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(_request, timeout=0):  # noqa: ANN001
        raise HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"rate_limited"}'),
        )

    monkeypatch.setattr("src.libs.llm.openai_llm.urlrequest.urlopen", fake_urlopen)

    llm = LLMFactory.create(
        {
            "llm": {
                "provider": "openai",
                "api_key": "sk-openai",
            }
        }
    )

    with pytest.raises(OpenAILLMError, match=r"provider=openai.*error_type=HTTPError"):
        llm.chat([{"role": "user", "content": "ping"}])
