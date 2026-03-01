"""Recursive Splitter 默认实现。"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from src.libs.splitter.base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """递归切分器实现。

    设计说明：
    1. 优先尝试使用 LangChain `RecursiveCharacterTextSplitter`。
    2. 若运行环境未安装 LangChain，则自动回退到本地 Markdown-aware 切分逻辑。
    3. 本地回退逻辑会优先保证“标题块 / 代码块”尽量不被截断。
    """

    def split_text(self, text: str, trace: Optional[Any] = None) -> List[str]:
        if not text:
            return []

        chunk_size = int(_read_setting(self.settings, "chunk_size", 500))
        chunk_overlap = int(_read_setting(self.settings, "chunk_overlap", 50))
        chunk_size = max(chunk_size, 1)
        chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

        langchain_chunks = _try_langchain_split(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if langchain_chunks is not None:
            return [chunk for chunk in langchain_chunks if chunk.strip()]

        return _fallback_markdown_aware_split(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def _try_langchain_split(text: str, chunk_size: int, chunk_overlap: int) -> Optional[List[str]]:
    """尝试使用 LangChain 切分；环境缺失依赖时返回 None。"""
    try:
        # 首选新版本导入路径（langchain-text-splitters 官方推荐）。
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        try:
            # 兼容部分版本中按模块拆分的导入路径。
            from langchain_text_splitters.character import RecursiveCharacterTextSplitter
        except Exception:
            try:
                # 兼容更早期的 langchain 主包导入路径。
                from langchain.text_splitter import RecursiveCharacterTextSplitter
            except Exception:
                return None

    splitter = RecursiveCharacterTextSplitter(
        # 参数选择说明：
        # 1) chunk_size/chunk_overlap 直接由配置驱动，保证与其它 splitter 行为一致。
        # 2) separators 从“段落 -> 行 -> 空格 -> 字符”逐级回退，尽量优先在语义边界切分。
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def _fallback_markdown_aware_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Markdown-aware 回退切分。

    实现策略：
    1. 先按“代码块/普通块”分段，确保 fenced code block 作为整体处理。
    2. 普通块再按段落拆分并聚合到 chunk_size。
    3. 超长单块才做滑窗切分；此时尽量只对普通文本切分。
    """
    blocks = _split_markdown_blocks(text)
    chunks: List[str] = []
    current = ""

    for block in blocks:
        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            current = ""

        if len(block) <= chunk_size:
            current = block
            continue

        # 对超长单块执行滑窗切分（例如极长段落）。
        chunks.extend(_sliding_window_split(block, chunk_size, chunk_overlap))

    if current.strip():
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


def _split_markdown_blocks(text: str) -> List[str]:
    """将 Markdown 文本切成“代码块”与“普通文本块”。"""
    code_pattern = re.compile(r"```.*?```", flags=re.DOTALL)
    blocks: List[str] = []
    cursor = 0

    for match in code_pattern.finditer(text):
        prefix = text[cursor : match.start()].strip()
        if prefix:
            blocks.extend(_split_paragraphs(prefix))

        code_block = match.group(0).strip()
        if code_block:
            blocks.append(code_block)
        cursor = match.end()

    tail = text[cursor:].strip()
    if tail:
        blocks.extend(_split_paragraphs(tail))

    return blocks


def _split_paragraphs(text: str) -> List[str]:
    """按空行切分段落，保留标题行在其段落内。"""
    parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return parts


def _sliding_window_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """对超长文本执行滑窗切分。"""
    if chunk_size <= 1:
        return [text]

    step = max(1, chunk_size - chunk_overlap)
    chunks: List[str] = []
    start = 0
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
