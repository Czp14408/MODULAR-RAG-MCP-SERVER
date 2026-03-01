"""B9: AzureVisionLLM 冒烟测试（mock HTTP）。"""

from pathlib import Path
import base64
import io
import json
import sys
from urllib.error import HTTPError, URLError

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.llm.azure_vision_llm import AzureVisionLLM
from src.libs.llm.base_vision_llm import VisionLLMError
from src.libs.llm.llm_factory import LLMFactory


class _FakeHTTPResponse:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _settings() -> dict:
    return {
        "vision_llm": {
            # 参数选择：使用 Azure 常见配置键，覆盖 endpoint/deployment/api_version。
            "provider": "azure",
            "endpoint": "https://example.openai.azure.com",
            "deployment_name": "gpt-4o-vision",
            "api_version": "2024-10-21",
            "api_key": "azure-secret",
            # 参数选择：设置较小值便于测试压缩分支是否触发。
            "max_image_size": 64,
            "timeout_seconds": 5,
        }
    }


def test_factory_creates_azure_vision_llm() -> None:
    vision_llm = LLMFactory.create_vision_llm(_settings())
    assert isinstance(vision_llm, AzureVisionLLM)
    print("[B9] factory routed to AzureVisionLLM")


def test_chat_with_image_path_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_file = tmp_path / "sample.png"
    image_file.write_bytes(b"fake-image-bytes")

    captured = {"url": "", "body": {}}

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse({"choices": [{"message": {"content": "vision-ok"}}]})

    monkeypatch.setattr("src.libs.llm.azure_vision_llm.urlrequest.urlopen", fake_urlopen)
    # 为了让测试不依赖 Pillow，压缩函数在此处替换为透传。
    monkeypatch.setattr(
        "src.libs.llm.azure_vision_llm.AzureVisionLLM._compress_image_if_needed",
        lambda self, image_bytes, max_image_size: image_bytes,
    )

    llm = AzureVisionLLM(_settings())
    response = llm.chat_with_image("describe it", str(image_file))

    print(f"[B9] request_url={captured['url']}")
    print(f"[B9] request_body={captured['body']}")
    print(f"[B9] response={response}")

    assert response.content == "vision-ok"
    assert "/openai/deployments/gpt-4o-vision/chat/completions?api-version=2024-10-21" in captured["url"]


def test_chat_with_base64_success(monkeypatch: pytest.MonkeyPatch) -> None:
    b64_img = base64.b64encode(b"raw-image").decode("utf-8")

    def fake_urlopen(_request, timeout=0):  # noqa: ANN001
        return _FakeHTTPResponse({"choices": [{"message": {"content": "from-base64"}}]})

    monkeypatch.setattr("src.libs.llm.azure_vision_llm.urlrequest.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "src.libs.llm.azure_vision_llm.AzureVisionLLM._compress_image_if_needed",
        lambda self, image_bytes, max_image_size: image_bytes,
    )

    llm = AzureVisionLLM(_settings())
    response = llm.chat_with_image("describe base64", b64_img)
    assert response.content == "from-base64"


def test_large_image_triggers_compress_hook(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_file = tmp_path / "big.png"
    image_file.write_bytes(b"big-image-bytes")

    called = {"max_image_size": 0}

    def fake_compress(self, image_bytes, max_image_size):  # noqa: ANN001
        called["max_image_size"] = max_image_size
        return b"compressed-image"

    def fake_urlopen(_request, timeout=0):  # noqa: ANN001
        return _FakeHTTPResponse({"choices": [{"message": {"content": "compressed-ok"}}]})

    monkeypatch.setattr(
        "src.libs.llm.azure_vision_llm.AzureVisionLLM._compress_image_if_needed",
        fake_compress,
    )
    monkeypatch.setattr("src.libs.llm.azure_vision_llm.urlrequest.urlopen", fake_urlopen)

    llm = AzureVisionLLM(_settings())
    response = llm.chat_with_image("compress me", str(image_file))

    print(f"[B9] compression_max_size={called['max_image_size']}")
    assert called["max_image_size"] == 64
    assert response.content == "compressed-ok"


def test_http_error_is_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(_request, timeout=0):  # noqa: ANN001
        raise HTTPError(
            url="https://example.openai.azure.com",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"code":"Unauthorized"}}'),
        )

    monkeypatch.setattr("src.libs.llm.azure_vision_llm.urlrequest.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "src.libs.llm.azure_vision_llm.AzureVisionLLM._compress_image_if_needed",
        lambda self, image_bytes, max_image_size: image_bytes,
    )

    llm = AzureVisionLLM(_settings())

    with pytest.raises(VisionLLMError, match=r"provider=azure_vision.*HTTPError.*401"):
        llm.chat_with_image("x", b"img")


def test_timeout_and_url_error_are_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout_urlopen(_request, timeout=0):  # noqa: ANN001
        raise TimeoutError("vision timeout")

    monkeypatch.setattr("src.libs.llm.azure_vision_llm.urlrequest.urlopen", timeout_urlopen)
    monkeypatch.setattr(
        "src.libs.llm.azure_vision_llm.AzureVisionLLM._compress_image_if_needed",
        lambda self, image_bytes, max_image_size: image_bytes,
    )

    llm = AzureVisionLLM(_settings())
    with pytest.raises(VisionLLMError, match=r"provider=azure_vision.*TimeoutError"):
        llm.chat_with_image("x", b"img")

    def urlerror_urlopen(_request, timeout=0):  # noqa: ANN001
        raise URLError("connection refused")

    monkeypatch.setattr("src.libs.llm.azure_vision_llm.urlrequest.urlopen", urlerror_urlopen)
    with pytest.raises(VisionLLMError, match=r"provider=azure_vision.*URLError"):
        llm.chat_with_image("x", b"img")
