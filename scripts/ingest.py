"""离线摄取脚本入口（C15）。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.settings import load_settings
from src.ingestion.pipeline import IngestionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline ingestion entrypoint")
    parser.add_argument("--collection", default="default", help="target collection name")
    parser.add_argument("--path", required=True, help="pdf file path to ingest")
    parser.add_argument("--config", default="config/settings.yaml", help="settings file path")
    parser.add_argument("--force", action="store_true", help="bypass integrity skip check")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    pipeline = IngestionPipeline(settings)
    result = pipeline.run(path=args.path, collection=args.collection, force=args.force)

    status = result.get("status", "unknown")
    print(f"[INGEST] status={status}")
    if status == "success":
        print(
            f"[INGEST] document_id={result.get('document_id')} "
            f"chunks={result.get('chunk_count')} "
            f"stored_images={result.get('stored_images')}"
        )
    elif status == "skipped":
        print(f"[INGEST] reason={result.get('reason')} file_hash={result.get('file_hash')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
