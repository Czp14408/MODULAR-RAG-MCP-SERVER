"""Ollama LLM provider（本地 HTTP 后端）。"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

from src.libs.llm.base_llm import BaseLLM, Message


class OllamaLLMError(RuntimeError):
    """Ollama provider 统一异常。"""


class OllamaLLM(BaseLLM):
    """通过 Ollama HTTP API 执行聊天补全。

    说明：
    1. 默认走 `POST {base_url}/api/chat`。
    2. 关闭流式返回（`stream=false`），便于统一抽取文本内容。
    3. 所有异常都包装成可读错误，并显式包含 provider 与错误类型。
    """

    provider_name = "ollama"

    def chat(self, messages: List[Message]) -> str:
        """调用 Ollama chat 接口并返回文本。"""
        normalized_messages = _normalize_messages(messages)

        payload = {
            "model": self._get_model_name(),
            "messages": normalized_messages,
            "stream": False,
        }

        req = self._build_request(payload)

        try:
            with urlrequest.urlopen(req, timeout=self._get_timeout_seconds()) as resp:
                body = resp.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
            raise OllamaLLMError(
                f"provider=ollama error_type=HTTPError status={exc.code} detail={detail}"
            ) from exc
        except urlerror.URLError as exc:
            raise OllamaLLMError(
                f"provider=ollama error_type=URLError detail={exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise OllamaLLMError(f"provider=ollama error_type=TimeoutError detail={exc}") from exc

        try:
            data = json.loads(body)
            message = data.get("message", {})
            content = message.get("content")
            if not isinstance(content, str):
                raise KeyError("missing message.content")
            return content
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaLLMError(
                f"provider=ollama error_type=ResponseParseError detail={exc}"
            ) from exc

    def _build_request(self, payload: Dict[str, Any]) -> urlrequest.Request:
        endpoint = f"{self._get_base_url().rstrip('/')}/api/chat"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        return urlrequest.Request(endpoint, data=data, headers=headers, method="POST")

    def _get_base_url(self) -> str:
        return str(_read_llm_option(self.settings, "base_url", default="http://localhost:11434"))

    def _get_model_name(self) -> str:
        model = _read_llm_option(self.settings, "model", default="llama3.1")
        return str(model)

    def _get_timeout_seconds(self) -> float:
        timeout = _read_llm_option(self.settings, "timeout_seconds", default=30)
        return float(timeout)


def _normalize_messages(messages: Sequence[Any]) -> List[Dict[str, str]]:
    """将输入消息转为 Ollama 所需的 `[{role, content}]` 格式。"""
    if not isinstance(messages, list) or not messages:
        raise OllamaLLMError(
            "provider=ollama error_type=ValidationError detail=messages must be non-empty list"
        )

    normalized: List[Dict[str, str]] = []
    for item in messages:
        if isinstance(item, Message):
            role, content = item.role, item.content
        elif is_dataclass(item):
            raw = asdict(item)
            role, content = raw.get("role"), raw.get("content")
        elif isinstance(item, dict):
            role, content = item.get("role"), item.get("content")
        else:
            raise OllamaLLMError(
                "provider=ollama error_type=ValidationError detail=invalid message item type"
            )

        if not isinstance(role, str) or not role.strip():
            raise OllamaLLMError(
                "provider=ollama error_type=ValidationError detail=message.role must be non-empty str"
            )
        if not isinstance(content, str):
            raise OllamaLLMError(
                "provider=ollama error_type=ValidationError detail=message.content must be str"
            )

        normalized.append({"role": role.strip(), "content": content})

    return normalized


def _read_llm_option(settings: Any, key: str, default: Any) -> Any:
    """从 settings.llm 读取配置，兼容 dataclass 与 dict。"""
    if hasattr(settings, "llm") and hasattr(settings.llm, key):
        value = getattr(settings.llm, key)
        return default if value is None else value
    if isinstance(settings, dict):
        llm = settings.get("llm")
        if isinstance(llm, dict):
            value = llm.get(key, default)
            return default if value is None else value
    return default
