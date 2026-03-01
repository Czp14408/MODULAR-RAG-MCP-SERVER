"""LLM Reranker：读取 prompt 并解析结构化排序输出。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.libs.llm.llm_factory import LLMFactory
from src.libs.reranker.base_reranker import (
    BaseReranker,
    RerankerContractError,
    RerankerFallbackSignal,
    _validate_candidates,
)


class LLMReranker(BaseReranker):
    """基于 LLM 的重排序实现。

    关键行为：
    1. 启动时加载 `config/prompts/rerank.txt`（可通过配置覆盖）。
    2. 调用 LLM 输出严格 JSON：`{"ranked_ids": ["id1", "id2", ...]}`。
    3. 若 LLM 调用失败或输出不合法，抛 `RerankerFallbackSignal`，交由上层回退。
    """

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        _validate_candidates(candidates)
        if not isinstance(query, str) or not query.strip():
            raise RerankerContractError("Invalid query: must be non-empty string")

        prompt_template = self._load_prompt_template()
        payload = self._build_payload(query=query, candidates=candidates, prompt_template=prompt_template)

        try:
            llm = LLMFactory.create(self.settings)
            llm_output = llm.chat([{"role": "user", "content": payload}])
            ranked_ids = self._parse_ranked_ids(llm_output)
        except RerankerContractError:
            # schema 错误明确上抛，便于调用方识别“输出不合规”。
            raise
        except Exception as exc:
            raise RerankerFallbackSignal(
                f"provider=llm_reranker error_type=LLMCallFailed detail={exc}"
            ) from exc

        id_to_candidate = {str(item.get("id")): item for item in candidates if "id" in item}
        missing = [doc_id for doc_id in ranked_ids if doc_id not in id_to_candidate]
        if missing:
            raise RerankerContractError(
                f"Invalid rerank output: ranked_ids contains unknown ids: {missing}"
            )

        seen = set()
        ranked: List[Dict[str, Any]] = []
        for doc_id in ranked_ids:
            if doc_id in seen:
                continue
            ranked.append(dict(id_to_candidate[doc_id]))
            seen.add(doc_id)

        # 没被模型覆盖到的候选，保留原顺序拼接在末尾，保证稳定性。
        for item in candidates:
            item_id = str(item.get("id"))
            if item_id not in seen:
                ranked.append(dict(item))

        return ranked

    def _load_prompt_template(self) -> str:
        """加载 rerank prompt；支持配置覆盖文本或路径。"""
        inline_prompt = _read_rerank_option(self.settings, "prompt_template", default=None)
        if isinstance(inline_prompt, str) and inline_prompt.strip():
            return inline_prompt

        prompt_path = _read_rerank_option(self.settings, "prompt_path", default="config/prompts/rerank.txt")
        path = Path(str(prompt_path))
        if not path.exists():
            raise RerankerFallbackSignal(
                f"provider=llm_reranker error_type=ConfigError detail=prompt file not found: {path}"
            )
        return path.read_text(encoding="utf-8")

    def _build_payload(self, query: str, candidates: List[Dict[str, Any]], prompt_template: str) -> str:
        """构造发给 LLM 的指令，要求严格返回 JSON。"""
        compact_candidates = [
            {
                "id": str(item.get("id", "")),
                "text": str(item.get("text", "")),
                "score": item.get("score"),
            }
            for item in candidates
        ]
        return (
            f"{prompt_template}\n\n"
            "Return strict JSON only with this schema: {\"ranked_ids\": [\"id1\", \"id2\"]}.\n"
            f"Query: {query}\n"
            f"Candidates: {json.dumps(compact_candidates, ensure_ascii=False)}"
        )

    def _parse_ranked_ids(self, llm_output: Any) -> List[str]:
        """解析 LLM 输出并校验 schema。"""
        if not isinstance(llm_output, str) or not llm_output.strip():
            raise RerankerContractError("Invalid rerank output: empty response")

        try:
            data = json.loads(llm_output)
        except json.JSONDecodeError as exc:
            raise RerankerContractError(f"Invalid rerank output: not valid JSON ({exc})") from exc

        ranked_ids = data.get("ranked_ids")
        if not isinstance(ranked_ids, list) or not ranked_ids:
            raise RerankerContractError("Invalid rerank output: ranked_ids must be non-empty list")

        normalized: List[str] = []
        for item in ranked_ids:
            if not isinstance(item, str) or not item.strip():
                raise RerankerContractError("Invalid rerank output: ranked_ids must contain strings")
            normalized.append(item.strip())
        return normalized


def _read_rerank_option(settings: Any, key: str, default: Any) -> Any:
    """从 settings.rerank 读取字段，兼容 dataclass 与 dict。"""
    if hasattr(settings, "rerank") and hasattr(settings.rerank, key):
        value = getattr(settings.rerank, key)
        return default if value is None else value
    if isinstance(settings, dict):
        rerank = settings.get("rerank")
        if isinstance(rerank, dict):
            value = rerank.get(key, default)
            return default if value is None else value
    return default
