"""C1: 核心类型契约测试。"""

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types import Chunk, ChunkRecord, Document


def _metadata() -> dict:
    return {
        "source_path": "data/documents/demo.pdf",
        "images": [
            {
                "id": "dochash_1_1",
                "path": "data/images/demo/dochash_1_1.png",
                "page": 1,
                "text_offset": 10,
                "text_length": 20,
                "position": {"x": 1, "y": 2, "w": 3, "h": 4},
            }
        ],
    }


def test_document_serialization_is_stable() -> None:
    doc = Document(id="doc-1", text="text with [IMAGE: dochash_1_1]", metadata=_metadata())
    data = doc.to_dict()
    restored = Document.from_dict(data)

    assert data == restored.to_dict()
    assert "[IMAGE: dochash_1_1]" in restored.text


def test_chunk_serialization_is_stable() -> None:
    chunk = Chunk(
        id="chunk-1",
        text="chunk text",
        metadata=_metadata(),
        start_offset=0,
        end_offset=10,
        source_ref="doc-1",
    )
    data = chunk.to_dict()
    restored = Chunk.from_dict(data)

    assert data == restored.to_dict()


def test_chunk_record_serialization_is_stable() -> None:
    record = ChunkRecord(
        id="chunk-1",
        text="chunk text",
        metadata=_metadata(),
        dense_vector=[0.1, 0.2],
        sparse_vector={"token": 1.0},
    )
    data = record.to_dict()
    restored = ChunkRecord.from_dict(data)

    assert data == restored.to_dict()


def test_source_path_is_required() -> None:
    with pytest.raises(ValueError, match="source_path"):
        Document(id="doc", text="x", metadata={})


def test_metadata_images_shape_validation() -> None:
    bad = _metadata()
    bad["images"] = [{"id": "", "path": "x"}]

    with pytest.raises(ValueError, match=r"images\[0\]\.id"):
        Document(id="doc", text="x", metadata=bad)
