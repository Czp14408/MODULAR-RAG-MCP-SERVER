"""E4: list_collections tool 测试。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mcp_server.tools.list_collections import list_collections


def test_list_collections_reads_directory_names(tmp_path: Path) -> None:
    documents_root = tmp_path / "data" / "documents"
    (documents_root / "demo").mkdir(parents=True)
    (documents_root / "finance").mkdir(parents=True)

    result = list_collections({}, {"documents_root": str(documents_root)})
    print(f"[E4] dir_result={result}")

    assert result["structuredContent"]["collections"] == ["demo", "finance"]


def test_list_collections_fallbacks_to_vector_store_metadata(tmp_path: Path) -> None:
    store_file = tmp_path / "store.json"
    store_file.write_text(
        json.dumps(
            [
                {"id": "1", "metadata": {"collection": "demo"}},
                {"id": "2", "metadata": {"collection": "ops"}},
            ]
        ),
        encoding="utf-8",
    )

    result = list_collections({}, {"documents_root": str(tmp_path / "missing"), "vector_store_file": str(store_file)})
    print(f"[E4] fallback_result={result}")

    assert result["structuredContent"]["collections"] == ["demo", "ops"]
