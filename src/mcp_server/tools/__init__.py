"""MCP tools 导出与 schema 注册。"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from src.mcp_server.tools.get_document_summary import get_document_summary
from src.mcp_server.tools.list_collections import list_collections
from src.mcp_server.tools.query_knowledge_hub import query_knowledge_hub


ToolHandler = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, object]]

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "query_knowledge_hub",
        "description": "Query the local knowledge hub and return cited markdown results.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
                "collection": {"type": "string"},
                "no_rerank": {"type": "boolean"},
            },
            "required": ["query"],
        },
        "handler": query_knowledge_hub,
    },
    {
        "name": "list_collections",
        "description": "List available collections.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
        "handler": list_collections,
    },
    {
        "name": "get_document_summary",
        "description": "Return title/summary/tags for a document.",
        "inputSchema": {
            "type": "object",
            "properties": {"doc_id": {"type": "string"}},
            "required": ["doc_id"],
        },
        "handler": get_document_summary,
    },
]

TOOL_REGISTRY: Dict[str, ToolHandler] = {item["name"]: item["handler"] for item in TOOLS}

__all__ = ["TOOLS", "TOOL_REGISTRY", "query_knowledge_hub", "list_collections", "get_document_summary"]
