"""PDF Loader 最小实现（C3）。"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


@dataclass
class _ExtractedImage:
    """内部图片载体：保存图片二进制和位置元数据。"""

    page: int
    image_index: int
    data: bytes
    position: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _PagePayload:
    """内部页面载体：每页文本 + 该页图片列表。"""

    text: str
    images: List[_ExtractedImage] = field(default_factory=list)


class PdfLoader(BaseLoader):
    """PDF Loader：抽取文本并按约定写入图片占位符与 metadata.images。"""

    def __init__(
        self,
        images_root: str = "data/images",
        image_placeholder_template: str = "[IMAGE: {image_id}]",
    ) -> None:
        # 参数选择说明：
        # 1) images_root 默认落在 data/images，便于后续 Pipeline 统一管理。
        # 2) placeholder 模板固定为 C1 规范要求的格式，确保上游/下游契约一致。
        self.images_root = Path(images_root)
        self.image_placeholder_template = image_placeholder_template

    def load(self, path: str) -> Document:
        pdf_path = Path(path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"pdf not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"PdfLoader only supports .pdf files: {pdf_path}")

        doc_hash = self._compute_doc_hash(pdf_path)
        page_payloads = self._extract_payloads(pdf_path)
        text, images_metadata = self._assemble_text_and_images(doc_hash, page_payloads)

        metadata: Dict[str, Any] = {"source_path": str(pdf_path)}
        if images_metadata:
            metadata["images"] = images_metadata
        return Document(id=doc_hash, text=text, metadata=metadata)

    def _compute_doc_hash(self, pdf_path: Path) -> str:
        # 参数选择说明：
        # 使用 SHA256 前 16 位作为 doc_id，兼顾稳定性与可读性。
        digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        return digest[:16]

    def _extract_payloads(self, pdf_path: Path) -> List[_PagePayload]:
        reader = self._open_reader(pdf_path)
        payloads: List[_PagePayload] = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = self._extract_page_text(page)
            page_images = self._extract_page_images(page, page_number)
            payloads.append(_PagePayload(text=page_text, images=page_images))
        return payloads

    def _open_reader(self, pdf_path: Path) -> Any:
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "PdfLoader requires dependency `pypdf`. "
                "Please install it via `pip install pypdf`."
            ) from exc
        return PdfReader(str(pdf_path))

    def _extract_page_text(self, page: Any) -> str:
        text = page.extract_text() or ""
        return text.strip()

    def _extract_page_images(self, page: Any, page_number: int) -> List[_ExtractedImage]:
        images: List[_ExtractedImage] = []
        try:
            page_images = list(getattr(page, "images", []) or [])
        except Exception as exc:  # noqa: BLE001
            # 图片抽取失败时按规范降级：记录 warning，但不阻塞文本解析。
            logger.warning("failed to read page.images on page=%s: %s", page_number, exc)
            return images

        for image_index, image in enumerate(page_images, start=1):
            try:
                raw_data = getattr(image, "data", b"")
                if callable(raw_data):
                    raw_data = raw_data()
                if not isinstance(raw_data, (bytes, bytearray)) or len(raw_data) == 0:
                    continue

                position = self._extract_image_position(image)
                images.append(
                    _ExtractedImage(
                        page=page_number,
                        image_index=image_index,
                        data=bytes(raw_data),
                        position=position,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                # 单张图片失败不影响同页其他图片与整文档解析。
                logger.warning(
                    "failed to extract image page=%s index=%s: %s",
                    page_number,
                    image_index,
                    exc,
                )
        return images

    def _extract_image_position(self, image: Any) -> Dict[str, Any]:
        # 参数选择说明：
        # C1 允许 position 可选；此处尽量提取 width/height，取不到则返回空 dict。
        position: Dict[str, Any] = {}
        for key in ("width", "height"):
            value = getattr(image, key, None)
            if isinstance(value, (int, float)):
                position[key] = value
        return position

    def _assemble_text_and_images(
        self,
        doc_hash: str,
        page_payloads: List[_PagePayload],
    ) -> tuple[str, List[Dict[str, Any]]]:
        text_parts: List[str] = []
        images_metadata: List[Dict[str, Any]] = []
        current_len = 0

        def append_text(piece: str) -> None:
            nonlocal current_len
            if not piece:
                return
            text_parts.append(piece)
            current_len += len(piece)

        for page_payload in page_payloads:
            if text_parts:
                append_text("\n\n")

            if page_payload.text:
                append_text(page_payload.text)

            for image in page_payload.images:
                if text_parts and not text_parts[-1].endswith("\n"):
                    append_text("\n")

                image_id = f"{doc_hash}_{image.page}_{image.image_index}"
                placeholder = self.image_placeholder_template.format(image_id=image_id)
                placeholder_offset = current_len
                append_text(placeholder)

                image_path = self._persist_image(doc_hash=doc_hash, image_id=image_id, data=image.data)
                image_meta: Dict[str, Any] = {
                    "id": image_id,
                    "path": image_path,
                    "page": image.page,
                    "text_offset": placeholder_offset,
                    "text_length": len(placeholder),
                }
                if image.position:
                    image_meta["position"] = image.position
                images_metadata.append(image_meta)

        return "".join(text_parts), images_metadata

    def _persist_image(self, doc_hash: str, image_id: str, data: bytes) -> str:
        image_dir = self.images_root / doc_hash
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{image_id}.png"
        image_path.write_bytes(data)
        return str(image_path)
