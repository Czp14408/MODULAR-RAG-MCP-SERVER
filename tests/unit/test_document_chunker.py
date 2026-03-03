"""C4: DocumentChunker 适配层测试。"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types import Document
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import SplitterFactory


class FakeSplitter(BaseSplitter):
    """测试假实现：固定返回两段文本，隔离第三方 splitter 细节。"""

    def split_text(self, text: str, trace: Optional[Any] = None) -> List[str]:
        if not text:
            return []
        return ["第一段内容", "第二段内容"]


def _build_document(text: str) -> Document:
    return Document(
        id="doc_demo",
        text=text,
        metadata={
            "source_path": "tests/data/test_chunking_text.pdf",
            "doc_type": "pdf",
            "title": "测试文档",
        },
    )


def test_document_chunker_converts_text_pieces_to_chunk_objects() -> None:
    SplitterFactory.register_provider("fake-chunker", FakeSplitter)
    chunker = DocumentChunker({"splitter": {"provider": "fake-chunker"}})
    document = _build_document("第一段内容\n\n第二段内容")

    chunks = chunker.split_document(document)
    print(f"[C4] fake splitter chunks_count={len(chunks)}")
    for c in chunks:
        print(f"[C4] chunk id={c.id} source_ref={c.source_ref} metadata={c.metadata}")

    assert len(chunks) == 2
    assert chunks[0].text == "第一段内容"
    assert chunks[1].text == "第二段内容"
    assert all(chunk.source_ref == document.id for chunk in chunks)
    assert all(chunk.metadata["source_path"] == document.metadata["source_path"] for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1]
    # 类型契约：Chunk 可序列化。
    assert chunks[0].to_dict()["id"] == chunks[0].id


def test_chunk_ids_are_unique_and_deterministic_for_same_document() -> None:
    SplitterFactory.register_provider("fake-chunker-stable", FakeSplitter)
    settings = {"splitter": {"provider": "fake-chunker-stable"}}
    document = _build_document("第一段内容\n\n第二段内容")

    chunker_1 = DocumentChunker(settings)
    chunker_2 = DocumentChunker(settings)
    chunks_1 = chunker_1.split_document(document)
    chunks_2 = chunker_2.split_document(document)

    ids_1 = [c.id for c in chunks_1]
    ids_2 = [c.id for c in chunks_2]
    print(f"[C4] deterministic ids run1={ids_1}")
    print(f"[C4] deterministic ids run2={ids_2}")

    assert len(set(ids_1)) == len(ids_1)
    assert ids_1 == ids_2
    assert ids_1[0].startswith("doc_demo_0000_")
    assert ids_1[1].startswith("doc_demo_0001_")


def test_chunk_count_changes_when_splitter_config_changes() -> None:
    # 参数选择说明：
    # 1) 使用 fixed splitter，避免外部依赖导致的不确定性。
    # 2) 同一文本分别使用 chunk_size=20/60，验证配置变化会影响 chunk 数量。
    text = "这是用于测试配置驱动切分行为的长文本。" * 10
    document = _build_document(text)

    small_cfg = {"splitter": {"provider": "fixed", "chunk_size": 20}}
    large_cfg = {"splitter": {"provider": "fixed", "chunk_size": 60}}

    chunks_small = DocumentChunker(small_cfg).split_document(document)
    chunks_large = DocumentChunker(large_cfg).split_document(document)

    print(
        f"[C4] config-driven counts small={len(chunks_small)} large={len(chunks_large)}"
    )

    assert len(chunks_small) > len(chunks_large)
    assert all(c.source_ref == document.id for c in chunks_small)
    assert all("chunk_index" in c.metadata for c in chunks_large)
