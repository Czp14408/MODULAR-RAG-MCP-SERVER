"""E1/E3/E6: MCP server 子进程集成测试。"""

from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path
import sys
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.embedding.hash_embedding import HashEmbedding


def _write_fixture_workspace(tmp_path: Path) -> Tuple[Path, Path]:
    """构造一个最小可运行的 MCP 工作区，供子进程 server 独立启动。"""
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "data" / "db" / "chroma").mkdir(parents=True)
    (tmp_path / "data" / "db" / "bm25").mkdir(parents=True)
    (tmp_path / "data" / "documents" / "demo").mkdir(parents=True)
    (tmp_path / "data" / "images").mkdir(parents=True)

    # 参数选择说明：
    # 这里固定使用 hash embedding + chroma/json 持久化，确保测试离线可复现。
    (tmp_path / "config" / "settings.yaml").write_text(
        "\n".join(
            [
                "llm:",
                "  provider: placeholder",
                "embedding:",
                "  provider: hash",
                "vector_store:",
                "  provider: chroma",
                "retrieval:",
                "  top_k: 5",
                "splitter:",
                "  provider: recursive",
                "  chunk_size: 120",
                "  chunk_overlap: 20",
                "rerank:",
                "  enabled: false",
                "evaluation:",
                "  enabled: false",
                "observability:",
                "  log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )

    image_path = tmp_path / "data" / "images" / "arch.png"
    image_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9sS2j7QAAAAASUVORK5CYII="
    )
    image_path.write_bytes(image_bytes)

    embedding = HashEmbedding({"embedding": {"provider": "hash"}}).embed(["分布式系统 数据库分片 架构设计"])[0]
    store_payload = [
        {
            "id": "chunk-demo-1",
            "vector": embedding,
            "text": "分布式系统通过水平分片提升扩展性，并结合服务治理提升稳定性。",
            "metadata": {
                "source_path": "data/documents/demo/architecture.pdf",
                "collection": "demo",
                "document_id": "demo-arch",
                "title": "现代分布式系统架构指南",
                "summary": "介绍系统架构与数据库分片策略。",
                "tags": ["distributed", "sharding"],
                "page": 1,
                "images": [
                    {
                        "id": "img-1",
                        "path": str(image_path),
                        "page": 1,
                        "text_offset": 0,
                        "text_length": 12,
                    }
                ],
            },
        }
    ]
    (tmp_path / "data" / "db" / "chroma" / "store.json").write_text(
        json.dumps(store_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    bm25_payload: Dict[str, object] = {
        "doc_count": 1,
        "avg_doc_length": 4.0,
        "terms": {
            "分布式系统": {
                "idf": -1.0986122886681098,
                "postings": [{"chunk_id": "chunk-demo-1", "tf": 1.0, "doc_length": 4}],
            },
            "数据库分片": {
                "idf": -1.0986122886681098,
                "postings": [{"chunk_id": "chunk-demo-1", "tf": 1.0, "doc_length": 4}],
            },
        },
        "_documents": {
            "chunk-demo-1": {
                "text": "分布式系统通过水平分片提升扩展性，并结合服务治理提升稳定性。",
                "metadata": {
                    "source_path": "data/documents/demo/architecture.pdf",
                    "collection": "demo",
                    "document_id": "demo-arch",
                    "title": "现代分布式系统架构指南",
                    "summary": "介绍系统架构与数据库分片策略。",
                    "tags": ["distributed", "sharding"],
                    "page": 1,
                    "images": [
                        {
                            "id": "img-1",
                            "path": str(image_path),
                            "page": 1,
                            "text_offset": 0,
                            "text_length": 12,
                        }
                    ],
                },
                "sparse_vector": {"分布式系统": 1.0, "数据库分片": 1.0},
            }
        },
    }
    (tmp_path / "data" / "db" / "bm25" / "bm25_index.json").write_text(
        json.dumps(bm25_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    server_script = PROJECT_ROOT / "src" / "mcp_server" / "server.py"
    return server_script, image_path


def _run_server_request(tmp_path: Path, request: Dict[str, object]) -> Tuple[Dict[str, object], str]:
    """启动一次独立子进程，发送单个 MCP 请求并读取 stdout/stderr。"""
    server_script, _ = _write_fixture_workspace(tmp_path)
    process = subprocess.run(
        [sys.executable, str(server_script)],
        input=json.dumps(request, ensure_ascii=False) + "\n",
        text=True,
        capture_output=True,
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        check=False,
    )

    stdout_lines = [line for line in process.stdout.splitlines() if line.strip()]
    assert stdout_lines, f"stdout is empty, stderr={process.stderr}"
    return json.loads(stdout_lines[0]), process.stderr


def test_mcp_server_initialize_uses_clean_stdout_and_logs_to_stderr(tmp_path: Path) -> None:
    response, stderr = _run_server_request(
        tmp_path,
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-01-01"}},
    )
    print(f"[E1] initialize_response={response}")
    print(f"[E1] stderr={stderr}")

    assert response["result"]["serverInfo"]["name"] == "modular-rag-mcp-server"
    assert "MCP server started" in stderr


def test_mcp_server_query_knowledge_hub_returns_cited_markdown(tmp_path: Path) -> None:
    response, stderr = _run_server_request(
        tmp_path,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "query_knowledge_hub",
                "arguments": {"query": "数据库分片", "collection": "demo", "top_k": 3, "no_rerank": True},
            },
        },
    )
    print(f"[E3] query_response={response}")
    print(f"[E3] query_stderr={stderr}")

    result = response["result"]
    assert result["content"][0]["type"] == "text"
    assert "[1]" in result["content"][0]["text"]
    citation = result["structuredContent"]["citations"][0]
    assert citation["source"] == "data/documents/demo/architecture.pdf"
    assert citation["chunk_id"] == "chunk-demo-1"


def test_mcp_server_query_returns_image_content_when_chunk_has_images(tmp_path: Path) -> None:
    response, _stderr = _run_server_request(
        tmp_path,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "query_knowledge_hub",
                "arguments": {"query": "分布式系统", "collection": "demo", "top_k": 3, "no_rerank": True},
            },
        },
    )
    print(f"[E6] image_response={response}")

    image_items = [item for item in response["result"]["content"] if item["type"] == "image"]
    assert image_items, "expected image content in MCP response"
    assert image_items[0]["mimeType"] == "image/png"
    assert isinstance(image_items[0]["data"], str)
    assert len(image_items[0]["data"]) > 10
