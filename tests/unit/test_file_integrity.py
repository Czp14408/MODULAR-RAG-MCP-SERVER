"""C2: FileIntegrity SQLite 实现测试。"""

from pathlib import Path
import sqlite3
import sys
import threading

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.loader.file_integrity import SQLiteIntegrityChecker


def test_compute_sha256_is_stable(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("same content", encoding="utf-8")

    checker = SQLiteIntegrityChecker(str(tmp_path / "ingestion_history.db"))
    h1 = checker.compute_sha256(str(sample))
    h2 = checker.compute_sha256(str(sample))
    print(f"[C2] SHA256 first={h1} second={h2}")

    assert h1 == h2


def test_mark_success_then_should_skip(tmp_path: Path) -> None:
    checker = SQLiteIntegrityChecker(str(tmp_path / "ingestion_history.db"))
    file_hash = "abc123"

    assert checker.should_skip(file_hash) is False
    checker.mark_success(file_hash=file_hash, file_path="/tmp/file.pdf", file_size=10, chunk_count=2)
    print(f"[C2] mark_success file_hash={file_hash}")
    assert checker.should_skip(file_hash) is True


def test_db_file_created_and_wal_enabled(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "db" / "ingestion_history.db"
    checker = SQLiteIntegrityChecker(str(db_path))

    assert db_path.exists()

    with sqlite3.connect(str(db_path)) as conn:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    print(f"[C2] sqlite journal_mode={mode} db_path={db_path}")
    assert str(mode).lower() == "wal"

    # 防止变量未使用警告并确保初始化完成
    assert checker is not None


def test_concurrent_writes_supported(tmp_path: Path) -> None:
    checker = SQLiteIntegrityChecker(str(tmp_path / "ingestion_history.db"))

    def worker(i: int) -> None:
        file_hash = f"hash-{i}"
        checker.mark_success(file_hash=file_hash, file_path=f"/tmp/{i}.pdf", file_size=i, chunk_count=i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with sqlite3.connect(str(tmp_path / "ingestion_history.db")) as conn:
        count = conn.execute("SELECT COUNT(*) FROM ingestion_history").fetchone()[0]
    print(f"[C2] concurrent insert count={count}")

    assert count == 20
