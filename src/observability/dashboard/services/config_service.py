"""ConfigService：读取并格式化当前系统配置。"""

from __future__ import annotations

from typing import Any, Dict


class ConfigService:
    """把 Settings/dict 配置转换为 Dashboard 友好的结构。"""

    def __init__(self, settings: Any) -> None:
        self.settings = settings

    def summarize(self) -> Dict[str, Dict[str, Any]]:
        return {
            "llm": {"provider": self._read(["llm", "provider"], "placeholder")},
            "embedding": {"provider": self._read(["embedding", "provider"], "unknown")},
            "vector_store": {"provider": self._read(["vector_store", "provider"], "unknown")},
            "splitter": {
                "provider": self._read(["splitter", "provider"], "recursive"),
                "chunk_size": self._read(["splitter", "chunk_size"], 500),
                "chunk_overlap": self._read(["splitter", "chunk_overlap"], 50),
            },
            "rerank": {"enabled": self._read(["rerank", "enabled"], False)},
            "evaluation": {"enabled": self._read(["evaluation", "enabled"], False)},
        }

    def _read(self, path: list[str], default: Any) -> Any:
        node: Any = self.settings
        for key in path:
            if isinstance(node, dict):
                node = node.get(key, default)
            elif hasattr(node, key):
                node = getattr(node, key)
            else:
                return default
        return node
