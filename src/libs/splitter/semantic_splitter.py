"""语义切分器的轻量占位实现。"""

from typing import Any, List, Optional

from src.libs.splitter.base_splitter import BaseSplitter


class SemanticSplitter(BaseSplitter):
    """按段落边界切分，模拟语义优先的切分行为。"""

    def split_text(self, text: str, trace: Optional[Any] = None) -> List[str]:
        if not text:
            return []
        # 用空行作为语义段落边界的简化策略。
        chunks = [part.strip() for part in text.split("\n\n") if part.strip()]
        return chunks if chunks else [text.strip()]
