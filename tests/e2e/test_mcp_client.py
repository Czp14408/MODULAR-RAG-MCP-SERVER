"""I1: MCP client 侧调用模拟。"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.embedding.hash_embedding import HashEmbedding


def _prepare_workspace(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "data" / "db" / "chroma").mkdir(parents=True)
    (tmp_path / "data" / "db" / "bm25").mkdir(parents=True)
    (tmp_path / "data" / "documents" / "demo").mkdir(parents=True)
    (tmp_path / "data" / "images").mkdir(parents=True)

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
    image_path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9sS2j7QAAAAASUVORK5CYII="
        )
    )
    embedding = HashEmbedding({"embedding": {"provider": "hash"}}).embed(["分布式系统 数据库分片"])[0]
    (tmp_path / "data" / "db" / "chroma" / "store.json").write_text(
        json.dumps(
            [
                {
                    "id": "chunk-demo-1",
                    "vector": embedding,
                    "text": "分布式系统通过水平分片提升扩展性。",
                    "metadata": {
                        "source_path": "data/documents/demo/architecture.pdf",
                        "collection": "demo",
                        "document_id": "demo-arch",
                        "title": "现代分布式系统架构指南",
                        "summary": "介绍系统架构与数据库分片策略。",
                        "page": 1,
                        "images": [{"id": "img-1", "path": str(image_path), "page": 1, "text_offset": 0, "text_length": 10}],
                    },
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "data" / "db" / "bm25" / "bm25_index.json").write_text(
        json.dumps(
            {
                "doc_count": 1,
                "avg_doc_length": 2.0,
                "terms": {
                    "分布式系统": {"idf": -1.0, "postings": [{"chunk_id": "chunk-demo-1", "tf": 1.0, "doc_length": 2}]},
                    "数据库分片": {"idf": -1.0, "postings": [{"chunk_id": "chunk-demo-1", "tf": 1.0, "doc_length": 2}]}
                },
                "_documents": {
                    "chunk-demo-1": {
                        "text": "分布式系统通过水平分片提升扩展性。",
                        "metadata": {
                            "source_path": "data/documents/demo/architecture.pdf",
                            "collection": "demo",
                            "document_id": "demo-arch",
                            "title": "现代分布式系统架构指南",
                            "summary": "介绍系统架构与数据库分片策略。"
                        },
                        "sparse_vector": {"分布式系统": 1.0, "数据库分片": 1.0}
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return PROJECT_ROOT / "src" / "mcp_server" / "server.py"


def test_mcp_client_can_initialize_and_query(tmp_path: Path) -> None:
    server_script = _prepare_workspace(tmp_path)
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-01-01"}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "query_knowledge_hub",
                "arguments": {"query": "数据库分片", "collection": "demo", "top_k": 3, "no_rerank": True},
            },
        },
    ]
    payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in requests) + "\n"

    process = subprocess.run(
        [sys.executable, str(server_script)],
        input=payload,
        text=True,
        capture_output=True,
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        check=False,
    )
    responses = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    print(f"[I1] responses={responses}")
    assert len(responses) == 3
    assert responses[0]["result"]["serverInfo"]["name"] == "modular-rag-mcp-server"
    assert responses[1]["result"]["tools"]
    assert responses[2]["result"]["structuredContent"]["citations"]
