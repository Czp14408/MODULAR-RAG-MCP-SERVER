"""MCP tool: query_knowledge_hub."""

from __future__ import annotations

from typing import Any, Dict

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.reranker import QueryReranker
from src.core.response.response_builder import ResponseBuilder


def query_knowledge_hub(arguments: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, object]:
    query = str(arguments.get("query", "")).strip()
    if not query:
        raise ValueError("query is required")

    top_k = int(arguments.get("top_k", 5))
    collection = str(arguments.get("collection", "")).strip()
    filters = {"collection": collection} if collection else None
    use_rerank = not bool(arguments.get("no_rerank", False))

    settings = context["settings"]
    hybrid = context.get("hybrid_search") or HybridSearch(settings)
    results = hybrid.search(query=query, top_k=top_k, filters=filters)

    if use_rerank:
        reranker = context.get("query_reranker") or QueryReranker(settings)
        results = reranker.rerank(query, results)

    builder = context.get("response_builder") or ResponseBuilder()
    return builder.build(results, query=query)
