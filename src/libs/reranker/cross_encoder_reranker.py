"""Cross-Encoder Reranker（Top-M 打分 + 回退信号）。"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.libs.reranker.base_reranker import BaseReranker, RerankerFallbackSignal, _validate_candidates


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder 重排序实现。

    关键行为：
    1. 仅对 Top-M 候选做打分，控制 CPU/GPU 成本。
    2. scorer 异常（超时/运行失败）时抛 `RerankerFallbackSignal`。
    3. 未参与打分的候选保持原顺序拼接到末尾。
    """

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        _validate_candidates(candidates)
        if not isinstance(query, str) or not query.strip():
            raise RerankerFallbackSignal(
                "provider=cross_encoder error_type=ValidationError detail=empty query"
            )

        top_m = int(_read_rerank_option(self.settings, "top_m", default=20))
        top_m = max(1, min(top_m, len(candidates)))

        head = [dict(item) for item in candidates[:top_m]]
        tail = [dict(item) for item in candidates[top_m:]]

        scorer = self._resolve_scorer()

        try:
            for item in head:
                text = str(item.get("text", ""))
                item["score"] = float(scorer(query, text))
        except TimeoutError as exc:
            raise RerankerFallbackSignal(
                f"provider=cross_encoder error_type=TimeoutError detail={exc}"
            ) from exc
        except Exception as exc:
            raise RerankerFallbackSignal(
                f"provider=cross_encoder error_type=ScorerError detail={exc}"
            ) from exc

        head.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return head + tail

    def _resolve_scorer(self) -> Callable[[str, str], float]:
        """解析 scorer：优先使用配置注入，否则走默认启发式评分。"""
        scorer = _read_rerank_option(self.settings, "scorer", default=None)
        if callable(scorer):
            return scorer
        return _default_overlap_score


def _default_overlap_score(query: str, text: str) -> float:
    """默认评分函数：按 query 与 text 的词项重叠率打分。"""
    query_terms = {term.strip().lower() for term in query.split() if term.strip()}
    text_terms = {term.strip().lower() for term in text.split() if term.strip()}
    if not query_terms:
        return 0.0
    return float(len(query_terms & text_terms)) / float(len(query_terms))


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
