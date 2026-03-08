"""MCP tool: query_knowledge_hub."""

from __future__ import annotations

from typing import Any, Dict

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.reranker import QueryReranker
from src.core.response.response_builder import ResponseBuilder
from src.core.trace import TraceCollector, TraceContext


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
    trace = TraceContext(trace_type="query")
    results = hybrid.search(query=query, top_k=top_k, filters=filters, trace=trace)

    if use_rerank:
        reranker = context.get("query_reranker") or QueryReranker(settings)
        results = reranker.rerank(query, results, trace=trace)

    builder = context.get("response_builder") or ResponseBuilder()
    payload = builder.build(results, query=query)
    TraceCollector().collect(trace)
    payload.setdefault("structuredContent", {})["trace_id"] = trace.trace_id
    return payload
