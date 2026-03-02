"""核心数据类型/契约（C1）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


Metadata = Dict[str, Any]
Vector = List[float]
SparseVector = Dict[str, float]


def _validate_source_path(metadata: Metadata, context: str) -> None:
    """所有核心对象的 metadata 最少必须包含 source_path。"""
    if not isinstance(metadata, dict):
        raise ValueError(f"{context}.metadata must be dict")
    source_path = metadata.get("source_path")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError(f"{context}.metadata.source_path is required")


def _validate_images(images: Any, context: str) -> None:
    """校验 metadata.images 契约。"""
    if images is None:
        return
    if not isinstance(images, list):
        raise ValueError(f"{context}.metadata.images must be list")

    for index, image in enumerate(images):
        if not isinstance(image, dict):
            raise ValueError(f"{context}.metadata.images[{index}] must be dict")

        image_id = image.get("id")
        image_path = image.get("path")
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError(f"{context}.metadata.images[{index}].id must be non-empty str")
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError(f"{context}.metadata.images[{index}].path must be non-empty str")

        # page/text_offset/text_length/position 允许缺失，但若存在需满足类型约束。
        if "page" in image and not isinstance(image["page"], int):
            raise ValueError(f"{context}.metadata.images[{index}].page must be int")
        if "text_offset" in image and not isinstance(image["text_offset"], int):
            raise ValueError(f"{context}.metadata.images[{index}].text_offset must be int")
        if "text_length" in image and not isinstance(image["text_length"], int):
            raise ValueError(f"{context}.metadata.images[{index}].text_length must be int")
        if "position" in image and not isinstance(image["position"], dict):
            raise ValueError(f"{context}.metadata.images[{index}].position must be dict")


@dataclass(frozen=True)
class Document:
    """文档对象契约。"""

    id: str
    text: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Document.id must be non-empty")
        if not isinstance(self.text, str):
            raise ValueError("Document.text must be str")
        _validate_source_path(self.metadata, "Document")
        _validate_images(self.metadata.get("images"), "Document")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class Chunk:
    """切片对象契约。"""

    id: str
    text: str
    metadata: Metadata
    start_offset: int
    end_offset: int
    source_ref: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Chunk.id must be non-empty")
        if not isinstance(self.text, str):
            raise ValueError("Chunk.text must be str")
        if not isinstance(self.start_offset, int) or not isinstance(self.end_offset, int):
            raise ValueError("Chunk offsets must be int")
        if self.start_offset < 0 or self.end_offset < self.start_offset:
            raise ValueError("Chunk offsets are invalid")
        _validate_source_path(self.metadata, "Chunk")
        _validate_images(self.metadata.get("images"), "Chunk")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": dict(self.metadata),
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            metadata=dict(data.get("metadata", {})),
            start_offset=int(data.get("start_offset", 0)),
            end_offset=int(data.get("end_offset", 0)),
            source_ref=data.get("source_ref"),
        )


@dataclass(frozen=True)
class ChunkRecord:
    """存储/检索载体契约。"""

    id: str
    text: str
    metadata: Metadata
    dense_vector: Optional[Vector] = None
    sparse_vector: Optional[SparseVector] = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("ChunkRecord.id must be non-empty")
        if not isinstance(self.text, str):
            raise ValueError("ChunkRecord.text must be str")
        _validate_source_path(self.metadata, "ChunkRecord")
        _validate_images(self.metadata.get("images"), "ChunkRecord")

        if self.dense_vector is not None:
            if not isinstance(self.dense_vector, list) or any(
                not isinstance(v, (int, float)) for v in self.dense_vector
            ):
                raise ValueError("ChunkRecord.dense_vector must be list[float]")

        if self.sparse_vector is not None:
            if not isinstance(self.sparse_vector, dict) or any(
                not isinstance(k, str) or not isinstance(v, (int, float))
                for k, v in self.sparse_vector.items()
            ):
                raise ValueError("ChunkRecord.sparse_vector must be dict[str, float]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": dict(self.metadata),
            "dense_vector": self.dense_vector,
            "sparse_vector": self.sparse_vector,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkRecord":
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            metadata=dict(data.get("metadata", {})),
            dense_vector=data.get("dense_vector"),
            sparse_vector=data.get("sparse_vector"),
        )


@dataclass(frozen=True)
class ProcessedQuery:
    """查询预处理后的结构化对象（预留给 D1）。"""

    query: str
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """检索结果对象（预留给 D2~D6）。"""

    id: str
    score: float
    text: str
    metadata: Metadata
