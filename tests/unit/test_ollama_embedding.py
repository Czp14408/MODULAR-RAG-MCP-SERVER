"""B7.4: Ollama Embedding provider 冒烟测试（mock HTTP）。"""

from pathlib import Path
import json
import sys
from urllib.error import URLError

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.ollama_embedding import OllamaEmbedding, OllamaEmbeddingError


class _FakeHTTPResponse:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_factory_routes_to_ollama_embedding() -> None:
    emb = EmbeddingFactory.create({"embedding": {"provider": "ollama"}})
    assert isinstance(emb, OllamaEmbedding)


def test_ollama_embedding_batch_success(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {"url": "", "body": {}}

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(
            {
                "embeddings": [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ]
            }
        )

    monkeypatch.setattr("src.libs.embedding.ollama_embedding.urlrequest.urlopen", fake_urlopen)

    emb = EmbeddingFactory.create(
        {
            "embedding": {
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "nomic-embed-text",
            }
        }
    )

    vectors = emb.embed(["alpha", "beta"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    assert captured["url"].endswith("/api/embed")
    assert captured["body"]["model"] == "nomic-embed-text"
    assert captured["body"]["input"] == ["alpha", "beta"]


def test_ollama_embedding_connection_error_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(_request, timeout=0):  # noqa: ANN001
        raise URLError("connection refused")

    monkeypatch.setattr("src.libs.embedding.ollama_embedding.urlrequest.urlopen", fake_urlopen)

    emb = EmbeddingFactory.create({"embedding": {"provider": "ollama"}})

    with pytest.raises(OllamaEmbeddingError, match=r"provider=ollama.*error_type=URLError"):
        emb.embed(["hello"])
