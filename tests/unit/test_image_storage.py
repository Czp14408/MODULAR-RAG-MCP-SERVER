"""C13: ImageStorage 测试。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.storage.image_storage import ImageStorage


def test_save_and_lookup_persisted_image_path(tmp_path: Path) -> None:
    storage = ImageStorage(
        images_root=str(tmp_path / "images"),
        db_path=str(tmp_path / "data" / "db" / "image_index.db"),
    )
    path = storage.save_image(
        image_id="img-1",
        image_bytes=b"fake-png-bytes",
        collection="demo",
        doc_hash="doc123",
        page_num=1,
    )
    found = storage.get_path("img-1")
    print(f"[C13] saved_path={path} found={found}")

    assert Path(path).exists()
    assert found == path


def test_mapping_is_persisted_in_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "db" / "image_index.db"
    storage = ImageStorage(images_root=str(tmp_path / "images"), db_path=str(db_path))
    storage.save_image("img-2", b"content", "demo", "docA", page_num=2)

    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT image_id, collection, doc_hash, page_num FROM image_index WHERE image_id = ?",
            ("img-2",),
        ).fetchone()
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    print(f"[C13] sqlite_row={row} journal_mode={mode}")

    assert row == ("img-2", "demo", "docA", 2)
    assert str(mode).lower() == "wal"


def test_list_by_collection_returns_expected_rows(tmp_path: Path) -> None:
    storage = ImageStorage(
        images_root=str(tmp_path / "images"),
        db_path=str(tmp_path / "data" / "db" / "image_index.db"),
    )
    storage.save_image("img-a", b"a", "demo", "doc1", page_num=1)
    storage.save_image("img-b", b"b", "demo", "doc2", page_num=2)
    storage.save_image("img-c", b"c", "other", "doc3", page_num=3)

    rows = storage.list_by_collection("demo")
    print(f"[C13] collection_rows={rows}")

    assert [row["image_id"] for row in rows] == ["img-a", "img-b"]
