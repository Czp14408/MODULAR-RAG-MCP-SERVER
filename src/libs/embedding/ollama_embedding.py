"""Ollama Embedding provider（本地 HTTP 后端）。"""

from __future__ import annotations

import json
from typing import Any, List
from urllib import error as urlerror
from urllib import request as urlrequest

from src.libs.embedding.base_embedding import BaseEmbedding


class OllamaEmbeddingError(RuntimeError):
    """Ollama Embedding 统一异常。"""


class OllamaEmbedding(BaseEmbedding):
    """通过 Ollama HTTP API 执行批量 embedding。

    说明：
    1. 默认调用 `POST {base_url}/api/embed`（支持批量 input）。
    2. 若服务返回单条 `embedding`，会在单输入场景自动包装成二维列表。
    3. 出错时输出可读错误，且不包含敏感配置。
    """

    provider_name = "ollama"

    def embed(self, texts: List[str], trace: Any = None) -> List[List[float]]:
        prepared = _prepare_texts(texts)

        payload = {
            "model": self._get_model_name(),
            "input": prepared,
        }

        req = self._build_request(payload)

        try:
            with urlrequest.urlopen(req, timeout=self._get_timeout_seconds()) as resp:
                body = resp.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
            raise OllamaEmbeddingError(
                f"provider=ollama error_type=HTTPError status={exc.code} detail={detail}"
            ) from exc
        except urlerror.URLError as exc:
            raise OllamaEmbeddingError(
                f"provider=ollama error_type=URLError detail={exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise OllamaEmbeddingError(
                f"provider=ollama error_type=TimeoutError detail={exc}"
            ) from exc

        try:
            data = json.loads(body)
            return _extract_embeddings(data, input_size=len(prepared))
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaEmbeddingError(
                f"provider=ollama error_type=ResponseParseError detail={exc}"
            ) from exc

    def _build_request(self, payload: dict) -> urlrequest.Request:
        endpoint = f"{self._get_base_url().rstrip('/')}/api/embed"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        return urlrequest.Request(endpoint, data=data, headers=headers, method="POST")

    def _get_base_url(self) -> str:
        return str(_read_embedding_option(self.settings, "base_url", default="http://localhost:11434"))

    def _get_model_name(self) -> str:
        model = _read_embedding_option(self.settings, "model", default="nomic-embed-text")
        return str(model)

    def _get_timeout_seconds(self) -> float:
        timeout = _read_embedding_option(self.settings, "timeout_seconds", default=30)
        return float(timeout)


def _prepare_texts(texts: List[str]) -> List[str]:
    """校验输入，要求非空 list[str]，且每项非空。"""
    if not isinstance(texts, list) or not texts:
        raise OllamaEmbeddingError(
            "provider=ollama error_type=ValidationError detail=texts must be non-empty list[str]"
        )

    prepared: List[str] = []
    for item in texts:
        if not isinstance(item, str):
            raise OllamaEmbeddingError(
                "provider=ollama error_type=ValidationError detail=text item must be str"
            )
        text = item.strip()
        if not text:
            raise OllamaEmbeddingError(
                "provider=ollama error_type=ValidationError detail=empty text is not allowed"
            )
        prepared.append(text)
    return prepared


def _extract_embeddings(data: dict, input_size: int) -> List[List[float]]:
    """解析 Ollama 响应，兼容 `embeddings` 与单条 `embedding` 两种形态。"""
    if "embeddings" in data:
        vectors_raw = data["embeddings"]
        if not isinstance(vectors_raw, list) or not vectors_raw:
            raise KeyError("missing embeddings")
    elif "embedding" in data and input_size == 1:
        vectors_raw = [data["embedding"]]
    else:
        raise KeyError("missing embeddings")

    vectors: List[List[float]] = []
    for vec in vectors_raw:
        if not isinstance(vec, list) or not vec:
            raise TypeError("invalid embedding vector")
        out_vec: List[float] = []
        for value in vec:
            if not isinstance(value, (int, float)):
                raise TypeError("embedding vector contains non-numeric value")
            out_vec.append(float(value))
        vectors.append(out_vec)

    return vectors


def _read_embedding_option(settings: Any, key: str, default: Any) -> Any:
    """从 settings.embedding 读取字段，兼容 dataclass 与 dict。"""
    if hasattr(settings, "embedding") and hasattr(settings.embedding, key):
        value = getattr(settings.embedding, key)
        return default if value is None else value
    if isinstance(settings, dict):
        embedding = settings.get("embedding")
        if isinstance(embedding, dict):
            value = embedding.get(key, default)
            return default if value is None else value
    return default
