"""B7.2: Ollama LLM provider 冒烟测试（mock HTTP）。"""

from pathlib import Path
import json
import sys
from urllib.error import URLError

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.llm.base_llm import Message
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.ollama_llm import OllamaLLM, OllamaLLMError


class _FakeHTTPResponse:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_factory_routes_to_ollama_provider() -> None:
    llm = LLMFactory.create({"llm": {"provider": "ollama"}})
    assert isinstance(llm, OllamaLLM)
    print("[B7.2] factory routed to OllamaLLM")


def test_ollama_chat_success_with_mock_http(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {"url": "", "body": {}}

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse({"message": {"content": "ollama-reply"}})

    monkeypatch.setattr("src.libs.llm.ollama_llm.urlrequest.urlopen", fake_urlopen)

    llm = LLMFactory.create(
        {
            "llm": {
                "provider": "ollama",
                # 参数选择：用默认本地地址，贴合开发机 Ollama 常见启动方式。
                "base_url": "http://localhost:11434",
                # 参数选择：给定具体模型，确保请求体中 model 字段可断言。
                "model": "llama3.1:8b",
            }
        }
    )

    result = llm.chat([Message(role="user", content="hello")])
    print(f"[B7.2] request_url={captured['url']}")
    print(f"[B7.2] request_body={captured['body']}")
    print(f"[B7.2] response_text={result}")

    assert result == "ollama-reply"
    assert captured["url"].endswith("/api/chat")
    assert captured["body"]["model"] == "llama3.1:8b"
    assert captured["body"]["stream"] is False


def test_ollama_connection_error_readable_and_no_sensitive_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(_request, timeout=0):  # noqa: ANN001
        raise URLError("connection refused")

    monkeypatch.setattr("src.libs.llm.ollama_llm.urlrequest.urlopen", fake_urlopen)

    llm = LLMFactory.create(
        {
            "llm": {
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "llama3.1",
                "api_key": "SHOULD_NOT_APPEAR",  # Ollama 不应依赖该配置
            }
        }
    )

    with pytest.raises(OllamaLLMError) as exc_info:
        llm.chat([{"role": "user", "content": "ping"}])

    msg = str(exc_info.value)
    assert "provider=ollama" in msg
    assert "error_type=URLError" in msg
    assert "SHOULD_NOT_APPEAR" not in msg
    print(f"[B7.2] error_message={msg}")
