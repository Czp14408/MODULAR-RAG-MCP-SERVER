"""QueryProcessor：关键词提取与 filters 结构化。"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.core.types import ProcessedQuery


class QueryProcessor:
    """对用户 query 做最小规则处理。"""

    _stopwords = {
        "的",
        "了",
        "和",
        "是",
        "在",
        "与",
        "the",
        "a",
        "an",
        "to",
        "for",
    }

    def process(self, query: str, filters: Optional[Dict[str, object]] = None) -> ProcessedQuery:
        if not isinstance(query, str) or not query.strip():
            return ProcessedQuery(query="", keywords=[], filters=dict(filters or {}))

        keywords = self._extract_keywords(query)
        return ProcessedQuery(query=query.strip(), keywords=keywords, filters=dict(filters or {}))

    def _extract_keywords(self, query: str) -> List[str]:
        tokens = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", query.lower())
        keywords = [token for token in tokens if token not in self._stopwords]
        return keywords or tokens[:1]
