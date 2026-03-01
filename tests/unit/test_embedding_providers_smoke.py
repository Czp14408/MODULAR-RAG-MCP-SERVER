"""B7.3: OpenAI & Azure Embedding provider 冒烟测试（mock HTTP）。"""

from pathlib import Path
import json
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.openai_embedding import OpenAIEmbeddingError


class _FakeHTTPResponse:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.mark.parametrize(
    "provider, cfg, expected_url_part",
    [
        (
            "openai",
            {
                "api_key": "sk-openai",
                "model": "text-embedding-3-small",
                "base_url": "https://api.openai.com/v1",
            },
            "/embeddings",
        ),
        (
            "azure",
            {
                "api_key": "azure-key",
                "endpoint": "https://example.openai.azure.com",
                "deployment": "text-embedding-ada-002",
                "api_version": "2024-10-21",
            },
            "/openai/deployments/text-embedding-ada-002/embeddings?api-version=2024-10-21",
        ),
    ],
)
def test_openai_and_azure_embedding_success(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    cfg: dict,
    expected_url_part: str,
) -> None:
    captured = {"url": "", "body": {}}

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(
            {
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                ]
            }
        )

    monkeypatch.setattr("src.libs.embedding.openai_embedding.urlrequest.urlopen", fake_urlopen)

    emb = EmbeddingFactory.create({"embedding": {"provider": provider, **cfg}})
    vectors = emb.embed(["alpha", "beta"])

    assert len(vectors) == 2
    assert all(len(v) == 3 for v in vectors)
    assert expected_url_part in captured["url"]


def test_empty_input_is_rejected() -> None:
    emb = EmbeddingFactory.create(
        {
            "embedding": {
                "provider": "openai",
                "api_key": "sk-openai",
            }
        }
    )
    with pytest.raises(OpenAIEmbeddingError, match="ValidationError"):
        emb.embed([])


def test_overlong_input_rejected_or_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    # 先验证默认策略：超长直接报错。
    emb_error = EmbeddingFactory.create(
        {
            "embedding": {
                "provider": "openai",
                "api_key": "sk-openai",
                "max_input_chars": 5,
                "truncate_input": False,
            }
        }
    )
    with pytest.raises(OpenAIEmbeddingError, match="text too long"):
        emb_error.embed(["123456789"])

    # 再验证截断策略：开启 truncate_input 后可继续请求。
    captured = {"input": []}

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        body = json.loads(request.data.decode("utf-8"))
        captured["input"] = body["input"]
        return _FakeHTTPResponse({"data": [{"index": 0, "embedding": [0.9, 0.1]}]})

    monkeypatch.setattr("src.libs.embedding.openai_embedding.urlrequest.urlopen", fake_urlopen)

    emb_ok = EmbeddingFactory.create(
        {
            "embedding": {
                "provider": "openai",
                "api_key": "sk-openai",
                "max_input_chars": 5,
                "truncate_input": True,
            }
        }
    )

    vectors = emb_ok.embed(["123456789"])
    assert vectors == [[0.9, 0.1]]
    assert captured["input"] == ["12345"]
