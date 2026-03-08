"""ImageStorage：图片文件存储 + SQLite 路径索引。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional


class ImageStorage:
    """保存图片文件并维护 image_id -> path 映射。"""

    def __init__(
        self,
        images_root: str = "data/images",
        db_path: str = "data/db/image_index.db",
    ) -> None:
        self.images_root = Path(images_root)
        self.images_root.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = Lock()
        self._init_db()

    def save_image(
        self,
        image_id: str,
        image_bytes: bytes,
        collection: str,
        doc_hash: str,
        page_num: Optional[int] = None,
        extension: str = ".png",
    ) -> str:
        if not image_id.strip():
            raise ValueError("image_id is required")
        if not image_bytes:
            raise ValueError("image_bytes is required")

        image_dir = self.images_root / collection
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{image_id}{extension}"
        image_path.write_bytes(image_bytes)

        with self._write_lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO image_index (image_id, file_path, collection, doc_hash, page_num)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(image_id) DO UPDATE SET
                        file_path=excluded.file_path,
                        collection=excluded.collection,
                        doc_hash=excluded.doc_hash,
                        page_num=excluded.page_num
                    """,
                    (image_id, str(image_path), collection, doc_hash, page_num),
                )
                conn.commit()
        return str(image_path)

    def get_path(self, image_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?",
                (image_id,),
            ).fetchone()
        return str(row[0]) if row else None

    def list_by_collection(self, collection: str) -> List[Dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT image_id, file_path, collection, doc_hash, page_num
                FROM image_index
                WHERE collection = ?
                ORDER BY image_id
                """,
                (collection,),
            ).fetchall()
        return [
            {
                "image_id": row[0],
                "file_path": row[1],
                "collection": row[2],
                "doc_hash": row[3],
                "page_num": row[4],
            }
            for row in rows
        ]

    def list_images(
        self,
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """按 collection/doc_hash 可选过滤列出图片。"""
        sql = """
            SELECT image_id, file_path, collection, doc_hash, page_num
            FROM image_index
            WHERE 1=1
        """
        params: List[object] = []
        if collection is not None:
            sql += " AND collection = ?"
            params.append(collection)
        if doc_hash is not None:
            sql += " AND doc_hash = ?"
            params.append(doc_hash)
        sql += " ORDER BY image_id"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "image_id": row[0],
                "file_path": row[1],
                "collection": row[2],
                "doc_hash": row[3],
                "page_num": row[4],
            }
            for row in rows
        ]

    def delete_by_doc_hash(
        self,
        doc_hash: str,
        collection: Optional[str] = None,
    ) -> int:
        """删除某文档关联图片及其索引。"""
        images = self.list_images(collection=collection, doc_hash=doc_hash)
        for image in images:
            file_path = Path(str(image["file_path"]))
            if file_path.exists():
                file_path.unlink()

        with self._connect() as conn:
            if collection is None:
                cursor = conn.execute("DELETE FROM image_index WHERE doc_hash = ?", (doc_hash,))
            else:
                cursor = conn.execute(
                    "DELETE FROM image_index WHERE doc_hash = ? AND collection = ?",
                    (doc_hash, collection),
                )
            conn.commit()
        return cursor.rowcount

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    collection TEXT,
                    doc_hash TEXT,
                    page_num INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash)")
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
