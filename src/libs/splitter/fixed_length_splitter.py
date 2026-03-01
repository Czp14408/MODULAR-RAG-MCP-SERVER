"""定长切分器实现。"""

from typing import Any, List, Optional

from src.libs.splitter.base_splitter import BaseSplitter


class FixedLengthSplitter(BaseSplitter):
    """严格按固定长度切分，不使用重叠。"""

    def split_text(self, text: str, trace: Optional[Any] = None) -> List[str]:
        if not text:
            return []
        chunk_size = int(_read_setting(self.settings, "chunk_size", 500))
        chunk_size = max(chunk_size, 1)

        chunks: List[str] = []
        for i in range(0, len(text), chunk_size):
            part = text[i : i + chunk_size].strip()
            if part:
                chunks.append(part)
        return chunks


def _read_setting(settings: Any, key: str, default: int) -> int:
    """读取 splitter 参数，优先 settings.splitter，再回退默认值。"""
    if hasattr(settings, "splitter") and hasattr(settings.splitter, key):
        return getattr(settings.splitter, key)
    if isinstance(settings, dict):
        splitter = settings.get("splitter")
        if isinstance(splitter, dict):
            return splitter.get(key, default)
    return default
