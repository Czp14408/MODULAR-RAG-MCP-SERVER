"""C15: ingest.py 命令行 e2e 测试。"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "ingest.py"


def test_ingest_script_produces_artifacts_and_skips_on_second_run(tmp_path: Path) -> None:
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

    first = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
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
    second = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
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

    print(f"[C15] first_stdout={first.stdout}")
    print(f"[C15] second_stdout={second.stdout}")
    print(f"[C15] first_stderr={first.stderr}")
    print(f"[C15] second_stderr={second.stderr}")

    assert first.returncode == 0
    assert second.returncode == 0
    assert "status=success" in first.stdout
    assert "status=skipped" in second.stdout
    assert (tmp_path / "data" / "db" / "chroma" / "store.json").exists()
    assert (tmp_path / "data" / "db" / "bm25" / "bm25_index.json").exists()
    assert (tmp_path / "data" / "db" / "ingestion_history.db").exists()
