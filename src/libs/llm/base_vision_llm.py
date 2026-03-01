"""Vision LLM 抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union


ImageInput = Union[str, bytes]


@dataclass(frozen=True)
class ChatResponse:
    """Vision LLM 返回对象。

    字段说明：
    - content: 主要文本输出
    - raw: 原始响应（便于追踪 provider-specific 字段）
    """

    content: str
    raw: Optional[dict] = None


class VisionLLMError(RuntimeError):
    """Vision LLM 通用错误。"""


class BaseVisionLLM(ABC):
    """多模态（文本+图片）LLM 抽象。

    设计目标：
    1. 将图片输入规范化（路径/bytes/base64）抽象为统一入口。
    2. 允许在接口层预留 `trace` 参数，便于后续接入链路追踪。
    3. 返回结构化 `ChatResponse`，避免后续扩展时破坏兼容性。
    """

    def __init__(self, settings: Any) -> None:
        # 保存配置对象，provider 实现按需读取自己的参数。
        self.settings = settings

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: ImageInput,
        trace: Optional[Any] = None,
    ) -> ChatResponse:
        """输入文本+图片并返回多模态理解结果。"""
