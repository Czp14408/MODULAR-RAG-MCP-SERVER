"""评估脚本入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.settings import load_settings
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.observability.evaluation.composite_evaluator import CompositeEvaluator
from src.observability.evaluation.eval_runner import EvalRunner
from src.observability.evaluation.ragas_evaluator import RagasEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation on golden test set")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--test-set", default="tests/fixtures/golden_test_set.json")
    parser.add_argument("--backend", choices=["custom", "ragas", "composite"], default="custom")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    hybrid_search = HybridSearch(settings)

    if args.backend == "custom":
        evaluator = CustomEvaluator(settings)
    elif args.backend == "ragas":
        evaluator = RagasEvaluator(settings)
    else:
        evaluator = CompositeEvaluator([CustomEvaluator(settings), RagasEvaluator(settings)], settings=settings)

    report = EvalRunner(settings, hybrid_search, evaluator).run(args.test_set)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
