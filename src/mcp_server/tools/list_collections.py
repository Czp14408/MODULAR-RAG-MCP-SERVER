"""MCP tool: list_collections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def list_collections(arguments: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, object]:
    documents_root = Path(str(context.get("documents_root", "data/documents")))
    collections = _list_from_documents_root(documents_root)
    if not collections:
        collections = _list_from_vector_store(context)

    names = sorted(collections)
    if not names:
        return {
            "content": [{"type": "text", "text": "当前没有可用集合。"}],
            "structuredContent": {"collections": []},
        }

    markdown = "\n".join(f"- {name}" for name in names)
    return {
        "content": [{"type": "text", "text": markdown}],
        "structuredContent": {"collections": names},
    }


def _list_from_documents_root(documents_root: Path) -> List[str]:
    if not documents_root.exists():
        return []
    return [item.name for item in documents_root.iterdir() if item.is_dir()]


def _list_from_vector_store(context: Dict[str, Any]) -> List[str]:
    store_file = Path(str(context.get("vector_store_file", "data/db/chroma/store.json")))
    if not store_file.exists():
        return []
    raw = json.loads(store_file.read_text(encoding="utf-8"))
    collections = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("collection"):
            collections.add(str(metadata["collection"]))
    return sorted(collections)
