"""DeepSeek provider（OpenAI-compatible 协议）。"""

from src.libs.llm.openai_llm import OpenAICompatibleLLM, _read_llm_option


class DeepSeekLLM(OpenAICompatibleLLM):
    """DeepSeek 实现。

    DeepSeek 对 OpenAI-compatible 协议支持较好，因此只需覆写默认配置。
    """

    provider_name = "deepseek"

    def _build_chat_endpoint(self) -> str:
        base_url = str(_read_llm_option(self.settings, "base_url", default="https://api.deepseek.com/v1"))
        return f"{base_url.rstrip('/')}/chat/completions"

    def _get_model_name(self) -> str:
        model = _read_llm_option(self.settings, "model", default="deepseek-chat")
        return str(model)
