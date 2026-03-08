"""ProtocolHandler：处理 MCP/JSON-RPC 基本方法。"""

from __future__ import annotations

from typing import Any, Dict

from src.mcp_server.tools import TOOLS, TOOL_REGISTRY


class ToolNotFoundError(ValueError):
    """当 tools/call 请求的 tool 未注册时抛出。"""


class ProtocolHandler:
    """解析 initialize/tools/list/tools/call 三类核心请求。"""

    def __init__(self, context: Dict[str, Any]) -> None:
        self.context = context

    def handle_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request_id = payload.get("id")
        method = payload.get("method")
        params = payload.get("params", {})

        if not isinstance(method, str) or not method:
            return self._error(request_id, -32600, "Invalid Request")
        if not isinstance(params, dict):
            return self._error(request_id, -32602, "Invalid params")

        try:
            if method == "initialize":
                return self._result(request_id, self.handle_initialize(params))
            if method == "tools/list":
                return self._result(request_id, self.handle_tools_list())
            if method == "tools/call":
                return self._result(request_id, self.handle_tools_call(params))
            return self._error(request_id, -32601, "Method not found")
        except ToolNotFoundError as exc:
            return self._error(request_id, -32601, str(exc))
        except ValueError as exc:
            return self._error(request_id, -32602, str(exc))
        except Exception:
            return self._error(request_id, -32603, "Internal error")

    def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        client_version = str(params.get("protocolVersion", "unknown"))
        return {
            "protocolVersion": client_version,
            "serverInfo": {
                "name": "modular-rag-mcp-server",
                "version": "0.1.0",
            },
            "capabilities": {"tools": {}},
        }

    def handle_tools_list(self) -> Dict[str, Any]:
        tools = [
            {
                "name": item["name"],
                "description": item["description"],
                "inputSchema": item["inputSchema"],
            }
            for item in TOOLS
        ]
        return {"tools": tools}

    def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tool name is required")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be object")

        handler = TOOL_REGISTRY.get(name)
        if handler is None:
            raise ToolNotFoundError(f"unknown tool: {name}")
        return handler(arguments, self.context)

    @staticmethod
    def _result(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
