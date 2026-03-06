"""BM25Indexer：倒排索引构建、持久化与查询。"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.core.types import ChunkRecord


class BM25Indexer:
    """构建并持久化 BM25 倒排索引。"""

    def __init__(self, persist_dir: str = "data/db/bm25") -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.persist_dir / "bm25_index.json"

    def build(self, records: Iterable[ChunkRecord]) -> Dict[str, dict]:
        """从稀疏记录构建索引并落盘。"""
        record_list = list(records)
        index = self._build_index(record_list)
        self._save(index)
        return index

    def update(self, records: Iterable[ChunkRecord]) -> Dict[str, dict]:
        """增量更新：与历史记录按 chunk_id 合并后重建。"""
        existing = self.load()
        existing_docs = existing.get("_documents", {})
        merged_docs = {
            chunk_id: {
                "text": item.get("text", ""),
                "metadata": item.get("metadata", {}),
                "sparse_vector": item.get("sparse_vector", {}),
            }
            for chunk_id, item in existing_docs.items()
        }

        for record in records:
            merged_docs[record.id] = {
                "text": record.text,
                "metadata": dict(record.metadata),
                "sparse_vector": dict(record.sparse_vector or {}),
            }

        rebuilt_records = [
            ChunkRecord(
                id=chunk_id,
                text=payload["text"],
                metadata=payload["metadata"],
                sparse_vector=payload["sparse_vector"],
            )
            for chunk_id, payload in merged_docs.items()
        ]
        return self.build(rebuilt_records)

    def load(self) -> Dict[str, dict]:
        if not self.index_file.exists():
            return {"doc_count": 0, "avg_doc_length": 0.0, "terms": {}, "_documents": {}}
        return json.loads(self.index_file.read_text(encoding="utf-8"))

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, object]]:
        """使用持久化索引执行 BM25 查询。"""
        index = self.load()
        terms = index.get("terms", {})
        docs = index.get("_documents", {})
        avg_doc_length = float(index.get("avg_doc_length", 0.0) or 0.0)
        if top_k <= 0 or not docs:
            return []

        scores: Dict[str, float] = defaultdict(float)
        query_terms = self._tokenize(query_text)
        for term in query_terms:
            term_entry = terms.get(term)
            if not isinstance(term_entry, dict):
                continue
            idf = float(term_entry.get("idf", 0.0))
            postings = term_entry.get("postings", [])
            for posting in postings:
                chunk_id = str(posting["chunk_id"])
                tf = float(posting["tf"])
                doc_length = int(posting["doc_length"])
                scores[chunk_id] += self._score_term(tf, doc_length, avg_doc_length, idf)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        results: List[Dict[str, object]] = []
        for chunk_id, score in ranked[:top_k]:
            doc = docs.get(chunk_id, {})
            results.append(
                {
                    "id": chunk_id,
                    "score": float(score),
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata", {}),
                }
            )
        return results

    def _build_index(self, records: List[ChunkRecord]) -> Dict[str, dict]:
        doc_count = len(records)
        doc_lengths: Dict[str, int] = {}
        doc_store: Dict[str, dict] = {}
        term_docs: Dict[str, List[dict]] = defaultdict(list)

        for record in records:
            sparse_vector = dict(record.sparse_vector or {})
            tokens = self._tokenize(record.text)
            doc_length = len(tokens)
            doc_lengths[record.id] = doc_length
            doc_store[record.id] = {
                "text": record.text,
                "metadata": dict(record.metadata),
                "sparse_vector": sparse_vector,
            }
            for term, tf in sparse_vector.items():
                term_docs[term].append(
                    {"chunk_id": record.id, "tf": float(tf), "doc_length": doc_length}
                )

        avg_doc_length = (
            sum(doc_lengths.values()) / float(doc_count)
            if doc_count > 0
            else 0.0
        )

        terms: Dict[str, dict] = {}
        for term, postings in term_docs.items():
            df = len(postings)
            # 参数选择说明：
            # 严格使用 spec 中给出的 BM25 IDF 公式，避免与后续检索阶段不一致。
            idf = math.log((doc_count - df + 0.5) / (df + 0.5)) if doc_count else 0.0
            terms[term] = {"idf": idf, "postings": postings}

        return {
            "doc_count": doc_count,
            "avg_doc_length": avg_doc_length,
            "terms": terms,
            "_documents": doc_store,
        }

    def _save(self, index: Dict[str, dict]) -> None:
        self.index_file.write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not isinstance(text, str) or not text.strip():
            return []
        return re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", text.lower())

    @staticmethod
    def _score_term(tf: float, doc_length: int, avg_doc_length: float, idf: float) -> float:
        # 参数选择说明：
        # 使用 BM25 常见默认参数 k1=1.5, b=0.75，作为当前阶段的稳定默认值。
        k1 = 1.5
        b = 0.75
        norm = 1 - b + b * (doc_length / avg_doc_length) if avg_doc_length > 0 else 1.0
        return idf * ((tf * (k1 + 1)) / (tf + k1 * norm))
