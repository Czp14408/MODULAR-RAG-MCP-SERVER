"""H5: Recall 回归测试。"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_recall_metrics_meet_threshold() -> None:
    for pdf_name in ["test_chunking_multimodal.pdf", "test_chunking_text.pdf"]:
        subprocess.run(
            [
                str(PROJECT_ROOT / ".venv" / "bin" / "python"),
                "scripts/ingest.py",
                "--collection",
                "demo",
                "--path",
                f"tests/data/{pdf_name}",
                "--force",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )

    result = subprocess.run(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "scripts/evaluate.py",
            "--backend",
            "custom",
            "--test-set",
            "tests/fixtures/golden_test_set.json",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    print(f"[H5] report={report}")

    assert report["metrics"]["hit_rate"] >= 0.5
    assert report["metrics"]["mrr"] >= 0.3
