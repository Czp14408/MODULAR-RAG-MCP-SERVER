"""Azure OpenAI provider（OpenAI-compatible 协议）。"""

from typing import Any

from src.libs.llm.openai_llm import OpenAICompatibleLLM, OpenAILLMError, _read_llm_option


class AzureLLM(OpenAICompatibleLLM):
    """Azure OpenAI 实现。

    与 OpenAI 官方接口差异：
    1. endpoint 路径包含 deployment 名称。
    2. 鉴权头通常使用 `api-key`，而不是 Bearer。
    3. 必须带 `api-version` 查询参数。
    """

    provider_name = "azure"

    def _build_chat_endpoint(self) -> str:
        endpoint = str(_read_llm_option(self.settings, "endpoint", default="")).rstrip("/")
        deployment = str(_read_llm_option(self.settings, "deployment", default=""))
        api_version = str(_read_llm_option(self.settings, "api_version", default="2024-02-01"))

        if not endpoint or not deployment:
            raise OpenAILLMError(
                "provider=azure error_type=ConfigError detail=missing endpoint or deployment"
            )

        return (
            f"{endpoint}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={api_version}"
        )

    def _build_request(self, payload: dict) -> Any:
        # 复写原因：Azure 使用 api-key 头部而不是 Authorization Bearer。
        from urllib import request as urlrequest
        import json

        api_key = self._get_api_key()
        if not api_key:
            raise OpenAILLMError("provider=azure error_type=ConfigError detail=missing api_key")

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }
        return urlrequest.Request(self._build_chat_endpoint(), data=data, headers=headers, method="POST")

    def _get_model_name(self) -> str:
        # Azure 场景下，model 字段使用 deployment 名称即可。
        deployment = _read_llm_option(self.settings, "deployment", default="")
        return str(deployment or "azure-deployment")
