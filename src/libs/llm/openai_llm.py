"""OpenAI-compatible LLM 实现基类与 OpenAI provider。"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

from src.libs.llm.base_llm import BaseLLM, Message


class OpenAILLMError(RuntimeError):
    """OpenAI provider 错误：消息中必须包含 provider 与错误类型。"""


class OpenAICompatibleLLM(BaseLLM):
    """OpenAI-compatible 协议的通用实现。

    设计要点：
    1. 将“消息校验、HTTP 请求、响应解析、错误包装”统一放在基类。
    2. 子类只需要覆盖 provider 名称和 endpoint 组装规则。
    3. 错误信息统一格式，便于日志检索和自动化排错。
    """

    provider_name: str = "openai-compatible"

    def chat(self, messages: List[Message]) -> str:
        """调用 OpenAI-compatible chat completions 接口并返回文本。"""
        normalized_messages = _normalize_messages(messages, provider=self.provider_name)

        payload = {
            "model": self._get_model_name(),
            "messages": normalized_messages,
        }

        # temperature 可选；配置存在时才附加，避免向部分 provider 发送不支持字段。
        temperature = _read_llm_option(self.settings, "temperature", default=None)
        if temperature is not None:
            payload["temperature"] = float(temperature)

        req = self._build_request(payload)

        try:
            with urlrequest.urlopen(req, timeout=self._get_timeout_seconds()) as resp:
                body = resp.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
            raise self._provider_error(
                f"provider={self.provider_name} error_type=HTTPError status={exc.code} detail={detail}"
            ) from exc
        except urlerror.URLError as exc:
            raise self._provider_error(
                f"provider={self.provider_name} error_type=URLError detail={exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise self._provider_error(
                f"provider={self.provider_name} error_type=TimeoutError detail={exc}"
            ) from exc

        try:
            data = json.loads(body)
            return _extract_content(data, provider=self.provider_name)
        except (ValueError, KeyError, TypeError) as exc:
            raise self._provider_error(
                f"provider={self.provider_name} error_type=ResponseParseError detail={exc}"
            ) from exc

    def _build_request(self, payload: Dict[str, Any]) -> urlrequest.Request:
        """构建 HTTP 请求对象。"""
        api_key = self._get_api_key()
        if not api_key:
            raise self._provider_error(
                f"provider={self.provider_name} error_type=ConfigError detail=missing api_key"
            )

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        return urlrequest.Request(self._build_chat_endpoint(), data=data, headers=headers, method="POST")

    def _provider_error(self, message: str) -> OpenAILLMError:
        """统一构造 provider 异常。"""
        return OpenAILLMError(message)

    def _build_chat_endpoint(self) -> str:
        """默认 OpenAI-compatible endpoint: {base_url}/chat/completions。"""
        base_url = str(_read_llm_option(self.settings, "base_url", default="https://api.openai.com/v1"))
        return f"{base_url.rstrip('/')}/chat/completions"

    def _get_model_name(self) -> str:
        model = _read_llm_option(self.settings, "model", default="gpt-4o-mini")
        return str(model)

    def _get_api_key(self) -> str:
        api_key = _read_llm_option(self.settings, "api_key", default="")
        return str(api_key)

    def _get_timeout_seconds(self) -> float:
        timeout = _read_llm_option(self.settings, "timeout_seconds", default=30)
        return float(timeout)


class OpenAILLM(OpenAICompatibleLLM):
    """OpenAI 官方 API provider。"""

    provider_name = "openai"


def _normalize_messages(messages: Sequence[Any], provider: str) -> List[Dict[str, str]]:
    """将输入消息标准化为 OpenAI-compatible 协议所需格式。

    允许两种输入：
    1. `Message` dataclass（项目内统一类型）
    2. dict: {"role": "...", "content": "..."}
    """
    if not isinstance(messages, list) or not messages:
        raise OpenAILLMError(
            f"provider={provider} error_type=ValidationError detail=messages must be non-empty list"
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
            raise OpenAILLMError(
                f"provider={provider} error_type=ValidationError detail=invalid message item type"
            )

        if not isinstance(role, str) or not role.strip():
            raise OpenAILLMError(
                f"provider={provider} error_type=ValidationError detail=message.role must be non-empty str"
            )
        if not isinstance(content, str):
            raise OpenAILLMError(
                f"provider={provider} error_type=ValidationError detail=message.content must be str"
            )

        normalized.append({"role": role.strip(), "content": content})

    return normalized


def _extract_content(data: Dict[str, Any], provider: str) -> str:
    """从 OpenAI-compatible 响应中提取文本内容。"""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise KeyError(f"provider={provider} missing choices")

    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str):
        raise KeyError(f"provider={provider} missing message.content")
    return content


def _read_llm_option(settings: Any, key: str, default: Any) -> Any:
    """从 settings 中读取 llm 下的字段，兼容 dataclass 和 dict 两种结构。"""
    if hasattr(settings, "llm") and hasattr(settings.llm, key):
        value = getattr(settings.llm, key)
        return default if value is None else value
    if isinstance(settings, dict):
        llm_settings = settings.get("llm")
        if isinstance(llm_settings, dict):
            value = llm_settings.get(key, default)
            return default if value is None else value
    return default
