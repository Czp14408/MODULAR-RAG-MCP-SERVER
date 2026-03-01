"""递归切分器的最小实现。"""

from typing import Any, List, Optional

from src.libs.splitter.base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """按固定窗口 + 重叠方式切分文本，作为递归策略占位实现。"""

    def split_text(self, text: str, trace: Optional[Any] = None) -> List[str]:
        if not text:
            return []

        chunk_size = int(_read_setting(self.settings, "chunk_size", 500))
        chunk_overlap = int(_read_setting(self.settings, "chunk_overlap", 50))
        chunk_size = max(chunk_size, 1)
        chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

        chunks: List[str] = []
        start = 0
        step = chunk_size - chunk_overlap
        while start < len(text):
            end = min(start + chunk_size, len(text))
            part = text[start:end].strip()
            if part:
                chunks.append(part)
            start += step
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
