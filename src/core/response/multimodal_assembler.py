"""MultimodalAssembler：将命中的图片组装为 MCP image content。"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, List

from src.core.types import RetrievalResult


class MultimodalAssembler:
    """读取 retrieval results 中的图片并转为 base64 content。"""

    def assemble(self, retrieval_results: List[RetrievalResult]) -> List[Dict[str, str]]:
        contents: List[Dict[str, str]] = []
        seen: set[str] = set()
        for item in retrieval_results:
            images = item.metadata.get("images", [])
            if not isinstance(images, list):
                continue
            for image in images:
                if not isinstance(image, dict):
                    continue
                image_path = Path(str(image.get("path", "")))
                if not image_path.exists():
                    continue
                image_key = str(image_path)
                if image_key in seen:
                    continue
                seen.add(image_key)
                mime = _guess_mime(image_path)
                contents.append(
                    {
                        "type": "image",
                        "mimeType": mime,
                        "data": base64.b64encode(image_path.read_bytes()).decode("utf-8"),
                    }
                )
        return contents


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".jpg" or suffix == ".jpeg":
        return "image/jpeg"
    if suffix == ".gif":
        return "image/gif"
    return "image/png"
