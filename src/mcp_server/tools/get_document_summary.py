"""MCP tool: get_document_summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def get_document_summary(arguments: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, object]:
    doc_id = str(arguments.get("doc_id", "")).strip()
    if not doc_id:
        raise ValueError("doc_id is required")

    store_file = Path(str(context.get("vector_store_file", "data/db/chroma/store.json")))
    if not store_file.exists():
        raise ValueError(f"document not found: {doc_id}")

    raw = json.loads(store_file.read_text(encoding="utf-8"))
    matched = _find_document_chunks(raw, doc_id)
    if not matched:
        raise ValueError(f"document not found: {doc_id}")

    first = matched[0]
    metadata = dict(first.get("metadata", {}))
    result = {
        "doc_id": _resolve_doc_id(metadata),
        "title": metadata.get("title", ""),
        "summary": metadata.get("summary", ""),
        "tags": metadata.get("tags", []),
        "source_path": metadata.get("source_path", ""),
    }
    return {
        "content": [
            {
                "type": "text",
                "text": f"{result['title']}\n\n{result['summary']}",
            }
        ],
        "structuredContent": result,
    }


def _find_document_chunks(raw: List[Dict[str, Any]], doc_id: str) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        if _resolve_doc_id(metadata) == doc_id:
            matched.append(item)
    return matched


def _resolve_doc_id(metadata: Dict[str, Any]) -> str:
    explicit = metadata.get("document_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    source_path = str(metadata.get("source_path", "")).strip()
    if source_path:
        return Path(source_path).stem
    return ""
