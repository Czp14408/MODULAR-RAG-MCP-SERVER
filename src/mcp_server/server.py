"""MCP server stdio entrypoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.settings import load_settings
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.reranker import QueryReranker
from src.core.response.response_builder import ResponseBuilder
from src.mcp_server.protocol_handler import ProtocolHandler
from src.observability.logger import get_logger


def build_context() -> Dict[str, Any]:
    settings = load_settings("config/settings.yaml")
    return {
        "settings": settings,
        "hybrid_search": HybridSearch(settings),
        "query_reranker": QueryReranker(settings),
        "response_builder": ResponseBuilder(),
        "documents_root": "data/documents",
        "vector_store_file": "data/db/chroma/store.json",
    }


def main() -> int:
    logger = get_logger(name="mcp_server")
    context = build_context()
    handler = ProtocolHandler(context)
    logger.info("MCP server started")

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request"},
            }
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
            continue

        response = handler.handle_request(payload)
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    logger.info("MCP server stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
