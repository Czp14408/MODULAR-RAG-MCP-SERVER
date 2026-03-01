"""Azure Vision LLM 实现。"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

from src.libs.llm.base_vision_llm import BaseVisionLLM, ChatResponse, ImageInput, VisionLLMError


class AzureVisionLLM(BaseVisionLLM):
    """通过 Azure OpenAI Vision 模型执行图像理解。

    支持：
    - 图片路径输入（str 且路径存在）
    - 图片 base64 输入（str 但路径不存在时视为 base64）
    - 图片 bytes 输入
    """

    provider_name = "azure"

    def chat_with_image(
        self,
        text: str,
        image_path: ImageInput,
        trace: Optional[Any] = None,
    ) -> ChatResponse:
        if not isinstance(text, str) or not text.strip():
            raise VisionLLMError(
                "provider=azure_vision error_type=ValidationError detail=text must be non-empty string"
            )

        image_b64 = self._prepare_image_base64(image_path)

        payload = {
            "model": self._get_deployment_name(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }
            ],
        }

        req = self._build_request(payload)

        try:
            with urlrequest.urlopen(req, timeout=self._get_timeout_seconds()) as resp:
                body = resp.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
            raise VisionLLMError(
                f"provider=azure_vision error_type=HTTPError status={exc.code} detail={detail}"
            ) from exc
        except urlerror.URLError as exc:
            raise VisionLLMError(
                f"provider=azure_vision error_type=URLError detail={exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise VisionLLMError(
                f"provider=azure_vision error_type=TimeoutError detail={exc}"
            ) from exc

        try:
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise KeyError("message.content missing")
            return ChatResponse(content=content, raw=data)
        except (ValueError, KeyError, TypeError) as exc:
            raise VisionLLMError(
                f"provider=azure_vision error_type=ResponseParseError detail={exc}"
            ) from exc

    def _prepare_image_base64(self, image_input: ImageInput) -> str:
        """统一处理图片输入，并在需要时执行尺寸压缩。"""
        raw_bytes = self._load_image_bytes(image_input)

        # 参数选择：默认 2048px，兼顾细节保留与请求成本控制。
        max_image_size = int(_read_vision_option(self.settings, "max_image_size", 2048))
        processed = self._compress_image_if_needed(raw_bytes, max_image_size=max_image_size)
        return base64.b64encode(processed).decode("utf-8")

    def _load_image_bytes(self, image_input: ImageInput) -> bytes:
        if isinstance(image_input, bytes):
            if not image_input:
                raise VisionLLMError(
                    "provider=azure_vision error_type=ValidationError detail=empty image bytes"
                )
            return image_input

        if isinstance(image_input, str):
            path = Path(image_input)
            if path.exists():
                return path.read_bytes()

            # 路径不存在时按 base64 字符串解析。
            try:
                return base64.b64decode(image_input, validate=True)
            except Exception as exc:
                raise VisionLLMError(
                    "provider=azure_vision error_type=ValidationError detail=image input is neither file path nor valid base64"
                ) from exc

        raise VisionLLMError(
            "provider=azure_vision error_type=ValidationError detail=image input must be str path/base64 or bytes"
        )

    def _compress_image_if_needed(self, image_bytes: bytes, max_image_size: int) -> bytes:
        """图片尺寸压缩：超出最大边时按比例缩放。

        说明：
        - 如果 Pillow 不可用，保守回退为原图（保证功能可用）。
        - 该函数是独立扩展点，便于后续引入更复杂压缩策略。
        """
        try:
            from PIL import Image
        except Exception:
            return image_bytes

        with Image.open(BytesIO(image_bytes)) as img:
            width, height = img.size
            longest = max(width, height)
            if longest <= max_image_size:
                return image_bytes

            ratio = float(max_image_size) / float(longest)
            new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
            resized = img.resize(new_size)
            out = BytesIO()
            resized.save(out, format="PNG")
            return out.getvalue()

    def _build_request(self, payload: dict) -> urlrequest.Request:
        api_key = self._get_api_key()
        if not api_key:
            raise VisionLLMError(
                "provider=azure_vision error_type=ConfigError detail=missing api_key"
            )

        endpoint = self._build_endpoint()
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }
        return urlrequest.Request(endpoint, data=data, headers=headers, method="POST")

    def _build_endpoint(self) -> str:
        endpoint = str(_read_vision_option(self.settings, "endpoint", "")).rstrip("/")
        deployment = self._get_deployment_name()
        api_version = str(_read_vision_option(self.settings, "api_version", "2024-02-01"))

        if not endpoint or not deployment:
            raise VisionLLMError(
                "provider=azure_vision error_type=ConfigError detail=missing endpoint or deployment_name"
            )

        return (
            f"{endpoint}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={api_version}"
        )

    def _get_api_key(self) -> str:
        return str(_read_vision_option(self.settings, "api_key", ""))

    def _get_deployment_name(self) -> str:
        return str(_read_vision_option(self.settings, "deployment_name", ""))

    def _get_timeout_seconds(self) -> float:
        return float(_read_vision_option(self.settings, "timeout_seconds", 30))


def _read_vision_option(settings: Any, key: str, default: Any) -> Any:
    """读取 vision_llm 配置，兼容 dataclass 与 dict。"""
    if hasattr(settings, "vision_llm") and hasattr(settings.vision_llm, key):
        value = getattr(settings.vision_llm, key)
        return default if value is None else value

    if isinstance(settings, dict):
        vision = settings.get("vision_llm")
        if isinstance(vision, dict):
            value = vision.get(key, default)
            return default if value is None else value

    return default
