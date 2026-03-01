"""OpenAI LLM provider stub used by early phases."""

from typing import List

from src.libs.llm.base_llm import BaseLLM, Message


class OpenAILLM(BaseLLM):
    """Minimal OpenAI provider implementation for factory wiring."""

    def chat(self, messages: List[Message]) -> str:
        if not messages:
            return ""
        return messages[-1].content
