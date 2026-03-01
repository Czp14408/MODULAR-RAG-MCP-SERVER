"""Base abstractions for pluggable LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Message:
    """单条对话消息。"""

    role: str
    content: str


class BaseLLM(ABC):
    """LLM 提供者无关的统一抽象接口。"""

    def __init__(self, settings) -> None:
        # 保留原始设置对象，便于子类读取 provider 特定配置。
        self.settings = settings

    @abstractmethod
    def chat(self, messages: List[Message]) -> str:
        """输入消息列表并返回模型文本响应。"""
