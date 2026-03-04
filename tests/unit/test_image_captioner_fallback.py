"""C7: ImageCaptioner 降级与启用路径测试。"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types import Chunk
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.libs.llm.base_vision_llm import BaseVisionLLM, ChatResponse


class FakeVisionLLM(BaseVisionLLM):
    def __init__(self, settings: Any, raise_error: bool = False) -> None:
        super().__init__(settings)
        self.raise_error = raise_error
        self.called = 0

    def chat_with_image(self, text: str, image_path: str, trace: Any = None) -> ChatResponse:
        self.called += 1
        if self.raise_error:
            raise RuntimeError("vision call failed")
        return ChatResponse(content=f"caption for {Path(image_path).name}")


def _chunk_with_image(path: str) -> Chunk:
    return Chunk(
        id="c-img",
        text="包含图片引用",
        metadata={
            "source_path": "tests/data/test_chunking_multimodal.pdf",
            "images": [{"id": "img1", "path": path, "page": 1, "text_offset": 0, "text_length": 10}],
        },
        start_offset=0,
        end_offset=6,
        source_ref="doc-img",
    )


def test_enabled_mode_generates_captions() -> None:
    img = PROJECT_ROOT / "tests" / "data" / "generated_assets" / "architecture_diagram.png"
    captioner = ImageCaptioner(
        settings={"ingestion": {"image_captioner": {"enabled": True, "prompt": "请描述图片"}}},
        vision_llm=FakeVisionLLM(settings={}),
    )
    out = captioner.caption([_chunk_with_image(str(img))])[0]
    print(f"[C7] enabled metadata={out.metadata}")

    assert out.metadata["images"][0]["caption"]
    assert "has_unprocessed_images" not in out.metadata


def test_disabled_mode_sets_unprocessed_flag_without_blocking() -> None:
    img = PROJECT_ROOT / "tests" / "data" / "generated_assets" / "architecture_diagram.png"
    captioner = ImageCaptioner(
        settings={"ingestion": {"image_captioner": {"enabled": False}}},
        vision_llm=FakeVisionLLM(settings={}),
    )
    out = captioner.caption([_chunk_with_image(str(img))])[0]
    print(f"[C7] disabled metadata={out.metadata}")

    assert out.metadata["has_unprocessed_images"] is True
    assert "caption" not in out.metadata["images"][0]


def test_error_mode_sets_unprocessed_flag_and_keeps_flow() -> None:
    img = PROJECT_ROOT / "tests" / "data" / "generated_assets" / "architecture_diagram.png"
    captioner = ImageCaptioner(
        settings={"ingestion": {"image_captioner": {"enabled": True}}},
        vision_llm=FakeVisionLLM(settings={}, raise_error=True),
    )
    out = captioner.caption([_chunk_with_image(str(img))])[0]
    print(f"[C7] error metadata={out.metadata}")

    assert out.metadata["has_unprocessed_images"] is True
    assert "caption_error" in out.metadata["images"][0]
