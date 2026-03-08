"""E2: ProtocolHandler 协议测试。"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mcp_server.protocol_handler import ProtocolHandler


def _build_handler() -> ProtocolHandler:
    """构造最小上下文，避免协议测试依赖完整检索链路。"""
    return ProtocolHandler(
        {
            "documents_root": "data/documents",
            "vector_store_file": "data/db/chroma/store.json",
        }
    )


def test_initialize_returns_server_info_and_capabilities() -> None:
    handler = _build_handler()

    response = handler.handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-01-01"}}
    )
    print(f"[E2] initialize_response={response}")

    assert response["result"]["serverInfo"]["name"] == "modular-rag-mcp-server"
    assert "tools" in response["result"]["capabilities"]


def test_tools_list_returns_registered_schemas() -> None:
    handler = _build_handler()

    response = handler.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    names = [item["name"] for item in response["result"]["tools"]]
    print(f"[E2] tool_names={names}")

    assert "query_knowledge_hub" in names
    assert "list_collections" in names
    assert "get_document_summary" in names


def test_invalid_method_returns_method_not_found() -> None:
    handler = _build_handler()

    response = handler.handle_request({"jsonrpc": "2.0", "id": 3, "method": "unknown", "params": {}})
    assert response["error"]["code"] == -32601


def test_unknown_tool_returns_method_not_found() -> None:
    handler = _build_handler()

    response = handler.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "missing_tool", "arguments": {}},
        }
    )
    print(f"[E2] unknown_tool_response={response}")

    assert response["error"]["code"] == -32601


def test_invalid_params_returns_invalid_params() -> None:
    handler = _build_handler()

    response = handler.handle_request(
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "", "arguments": {}}}
    )
    assert response["error"]["code"] == -32602


def test_internal_error_does_not_leak_stack_trace() -> None:
    """通过目录读文件错误触发非 ValueError 异常，确认只返回统一错误文案。"""
    handler = ProtocolHandler({"vector_store_file": str(Path(__file__).parent)})

    response = handler.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "get_document_summary", "arguments": {"doc_id": "doc-1"}},
        }
    )
    print(f"[E2] internal_error_response={response}")

    assert response["error"]["code"] == -32603
    assert response["error"]["message"] == "Internal error"
