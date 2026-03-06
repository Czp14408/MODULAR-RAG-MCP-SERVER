"""C11: BM25Indexer roundtrip 测试。"""

from __future__ import annotations

import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types import ChunkRecord
from src.ingestion.storage.bm25_indexer import BM25Indexer


def _record(chunk_id: str, text: str, sparse: dict) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        text=text,
        metadata={"source_path": "tests/data/test_chunking_text.pdf"},
        sparse_vector=sparse,
    )


def test_build_load_and_query_return_stable_top_ids(tmp_path: Path) -> None:
    indexer = BM25Indexer(persist_dir=str(tmp_path / "bm25"))
    records = [
        _record("c1", "rag 检索 系统", {"rag": 0.4, "检索": 0.4, "系统": 0.2}),
        _record("c2", "数据库 分片", {"数据库": 0.5, "分片": 0.5}),
        _record("c3", "rag 向量 检索", {"rag": 0.3, "向量": 0.3, "检索": 0.4}),
    ]

    indexer.build(records)
    loaded = indexer.load()
    result_1 = indexer.query("rag 检索", top_k=2)
    result_2 = indexer.query("rag 检索", top_k=2)

    print(f"[C11] loaded_terms={list(loaded['terms'].keys())}")
    print(f"[C11] result_1={result_1}")
    print(f"[C11] result_2={result_2}")

    assert loaded["doc_count"] == 3
    assert [item["id"] for item in result_1] == [item["id"] for item in result_2]
    assert result_1[0]["id"] in {"c1", "c3"}


def test_idf_formula_matches_spec(tmp_path: Path) -> None:
    indexer = BM25Indexer(persist_dir=str(tmp_path / "bm25"))
    records = [
        _record("c1", "alpha beta", {"alpha": 0.5, "beta": 0.5}),
        _record("c2", "alpha gamma", {"alpha": 0.5, "gamma": 0.5}),
        _record("c3", "beta gamma", {"beta": 0.5, "gamma": 0.5}),
    ]

    built = indexer.build(records)
    actual = built["terms"]["alpha"]["idf"]
    expected = math.log((3 - 2 + 0.5) / (2 + 0.5))
    print(f"[C11] alpha_idf actual={actual} expected={expected}")

    assert abs(actual - expected) < 1e-9


def test_update_supports_incremental_rebuild(tmp_path: Path) -> None:
    indexer = BM25Indexer(persist_dir=str(tmp_path / "bm25"))
    indexer.build([_record("c1", "alpha", {"alpha": 1.0})])
    updated = indexer.update([_record("c2", "beta", {"beta": 1.0})])
    print(f"[C11] updated_doc_count={updated['doc_count']} terms={list(updated['terms'])}")

    assert updated["doc_count"] == 2
    assert "beta" in updated["terms"]
