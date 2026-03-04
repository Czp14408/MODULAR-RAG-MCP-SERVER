"""ImageCaptioner：可选图片描述生成，失败降级不阻塞。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Optional

from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_vision_llm import BaseVisionLLM
from src.libs.llm.llm_factory import LLMFactory


class ImageCaptioner(BaseTransform):
    """对 chunk 中图片引用生成 caption。"""

    def __init__(self, settings: Any, vision_llm: Optional[BaseVisionLLM] = None) -> None:
        self.settings = settings
        self.enabled = self._read_enabled(settings)
        self.prompt = self._read_prompt(settings)
        self._vision_llm: Optional[BaseVisionLLM] = vision_llm
        self._init_error: Optional[str] = None

        if self.enabled and self._vision_llm is None:
            try:
                self._vision_llm = LLMFactory.create_vision_llm(settings)
            except Exception as exc:  # noqa: BLE001
                self._init_error = str(exc)
                self._vision_llm = None

    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        return self.caption(chunks, trace=trace)

    def caption(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        output: List[Chunk] = []
        captioned_count = 0
        unprocessed_count = 0

        for chunk in chunks:
            metadata = dict(chunk.metadata)
            images = self._extract_images(metadata)
            if not images:
                output.append(chunk)
                continue

            if not self.enabled or self._vision_llm is None:
                metadata["has_unprocessed_images"] = True
                if self._init_error:
                    metadata["image_caption_fallback_reason"] = self._init_error
                output.append(replace(chunk, metadata=metadata))
                unprocessed_count += 1
                continue

            new_images = []
            for image in images:
                image_meta = dict(image)
                image_path = str(image_meta.get("path", ""))
                try:
                    response = self._vision_llm.chat_with_image(text=self.prompt, image_path=image_path)
                    image_meta["caption"] = response.content.strip()
                    captioned_count += 1
                except Exception as exc:  # noqa: BLE001
                    image_meta["caption_error"] = str(exc)
                    metadata["has_unprocessed_images"] = True
                new_images.append(image_meta)

            metadata["images"] = new_images
            output.append(replace(chunk, metadata=metadata))

        if trace is not None:
            trace.record_stage(
                "image_captioner",
                elapsed_ms=0.0,
                chunk_count=len(chunks),
                enabled=self.enabled,
                captioned_count=captioned_count,
                unprocessed_count=unprocessed_count,
            )
        return output

    @staticmethod
    def _extract_images(metadata: dict) -> List[dict]:
        # 同时兼容 C1 的 metadata.images 与后续可能出现的 image_refs。
        images = metadata.get("images")
        if isinstance(images, list):
            return [item for item in images if isinstance(item, dict)]
        image_refs = metadata.get("image_refs")
        if isinstance(image_refs, list):
            return [item for item in image_refs if isinstance(item, dict)]
        return []

    @staticmethod
    def _read_enabled(settings: Any) -> bool:
        if isinstance(settings, dict):
            ingestion = settings.get("ingestion", {})
            if isinstance(ingestion, dict):
                captioner = ingestion.get("image_captioner", {})
                if isinstance(captioner, dict):
                    return bool(captioner.get("enabled", False))
        if hasattr(settings, "ingestion") and hasattr(settings.ingestion, "image_captioner"):
            node = settings.ingestion.image_captioner
            if hasattr(node, "enabled"):
                return bool(node.enabled)
        return False

    @staticmethod
    def _read_prompt(settings: Any) -> str:
        default = "请用一句话描述这张图片的主要内容。"
        if isinstance(settings, dict):
            ingestion = settings.get("ingestion", {})
            if isinstance(ingestion, dict):
                captioner = ingestion.get("image_captioner", {})
                if isinstance(captioner, dict):
                    value = captioner.get("prompt", default)
                    return str(value)
        if hasattr(settings, "ingestion") and hasattr(settings.ingestion, "image_captioner"):
            node = settings.ingestion.image_captioner
            if hasattr(node, "prompt"):
                return str(node.prompt)
        return default
