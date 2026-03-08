"""D7: query.py CLI e2e 测试。"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INGEST_SCRIPT = PROJECT_ROOT / "scripts" / "ingest.py"
QUERY_SCRIPT = PROJECT_ROOT / "scripts" / "query.py"


def test_query_script_returns_formatted_results_and_verbose_output(tmp_path: Path) -> None:
    pdf_path = PROJECT_ROOT / "tests" / "data" / "test_chunking_multimodal.pdf"
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm:",
                "  provider: placeholder",
                "embedding:",
                "  provider: hash",
                "vector_store:",
                "  provider: chroma",
                "retrieval:",
                "  top_k: 5",
                "splitter:",
                "  provider: recursive",
                "  chunk_size: 120",
                "  chunk_overlap: 20",
                "ingestion:",
                "  chunk_refiner:",
                "    use_llm: false",
                "rerank:",
                "  enabled: false",
                "evaluation:",
                "  enabled: false",
                "observability:",
                "  log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )

    ingest = subprocess.run(
        [
            sys.executable,
            str(INGEST_SCRIPT),
            "--collection",
            "demo",
            "--path",
            str(pdf_path),
            "--config",
            str(config_path),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    query = subprocess.run(
        [
            sys.executable,
            str(QUERY_SCRIPT),
            "--query",
            "分布式 分片",
            "--top-k",
            "2",
            "--collection",
            "demo",
            "--config",
            str(config_path),
            "--verbose",
            "--no-rerank",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )

    print(f"[D7] ingest_stdout={ingest.stdout}")
    print(f"[D7] query_stdout={query.stdout}")
    print(f"[D7] query_stderr={query.stderr}")

    assert ingest.returncode == 0
    assert query.returncode == 0
    assert "[VERBOSE] keywords=" in query.stdout
    assert "score=" in query.stdout
