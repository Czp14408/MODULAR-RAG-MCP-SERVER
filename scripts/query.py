"""查询命令行入口（D7）。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional
import warnings

try:
    from urllib3.exceptions import NotOpenSSLWarning  # type: ignore

    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.reranker import QueryReranker
from src.core.settings import load_settings
from src.core.trace import TraceCollector, TraceContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query local knowledge base")
    parser.add_argument("--query", required=True, help="query text")
    parser.add_argument("--top-k", type=int, default=10, help="number of results to return")
    parser.add_argument("--collection", default="", help="optional collection filter")
    parser.add_argument("--config", default="config/settings.yaml", help="settings file path")
    parser.add_argument("--verbose", action="store_true", help="print intermediate stages")
    parser.add_argument("--no-rerank", action="store_true", help="skip rerank stage")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)

    filters = {"collection": args.collection} if args.collection else None
    hybrid = HybridSearch(settings)
    trace = TraceContext(trace_type="query")
    results = hybrid.search(query=args.query, top_k=args.top_k, filters=filters, trace=trace)

    if not results:
        TraceCollector().collect(trace)
        print("未找到相关文档，请先运行 ingest.py 摄取数据。")
        return 0

    if not args.no_rerank:
        reranker = QueryReranker(settings)
        results = reranker.rerank(args.query, results, trace=trace)

    TraceCollector().collect(trace)

    if args.verbose:
        _print_verbose(hybrid)

    for idx, item in enumerate(results, start=1):
        source = item.metadata.get("source_path", "unknown")
        page = item.metadata.get("page", "-")
        preview = item.text[:100].replace("\n", " ")
        print(f"{idx}. score={item.score:.4f} source={source} page={page}")
        print(f"   {preview}")
    return 0


def _print_verbose(hybrid: HybridSearch) -> None:
    debug = hybrid.last_debug
    processed = debug.get("processed_query")
    if processed is not None:
        print(f"[VERBOSE] keywords={processed.keywords} filters={processed.filters}")
    print(f"[VERBOSE] dense_count={len(debug.get('dense_results', []))}")
    print(f"[VERBOSE] sparse_count={len(debug.get('sparse_results', []))}")
    print(f"[VERBOSE] fusion_count={len(debug.get('fusion_results', []))}")
    if debug.get("dense_error"):
        print(f"[VERBOSE] dense_error={debug['dense_error']}")
    if debug.get("sparse_error"):
        print(f"[VERBOSE] sparse_error={debug['sparse_error']}")


if __name__ == "__main__":
    raise SystemExit(main())
