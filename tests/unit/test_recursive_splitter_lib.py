"""B7.5: Recursive Splitter 默认实现测试。"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.splitter.recursive_splitter import RecursiveSplitter
from src.libs.splitter.splitter_factory import SplitterFactory


def test_factory_creates_recursive_splitter() -> None:
    splitter = SplitterFactory.create({"splitter": {"provider": "recursive", "chunk_size": 120}})
    assert isinstance(splitter, RecursiveSplitter)


def test_recursive_splitter_handles_markdown_heading_and_code_block() -> None:
    markdown = (
        "# 标题\n\n"
        "这是第一段说明文本。\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "这是代码块后的补充说明。"
    )

    splitter = RecursiveSplitter({"splitter": {"chunk_size": 80, "chunk_overlap": 10}})
    chunks = splitter.split_text(markdown)

    assert chunks
    joined = "\n\n".join(chunks)
    # 标题内容应保留。
    assert "# 标题" in joined
    # 代码块边界应完整存在，不应把 ``` 打散。
    assert "```python" in joined
    assert "return a + b" in joined
    assert "```" in joined
