"""文件完整性检查（C2）。"""

from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional


class FileIntegrityChecker(ABC):
    """文件完整性检查抽象接口。"""

    @abstractmethod
    def compute_sha256(self, path: str) -> str:
        """计算文件 SHA256。"""

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """若文件已成功处理则返回 True。"""

    @abstractmethod
    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        file_size: Optional[int] = None,
        chunk_count: Optional[int] = None,
    ) -> None:
        """标记文件处理成功。"""

    @abstractmethod
    def mark_failed(
        self,
        file_hash: str,
        error_msg: str,
        file_path: str = "",
        file_size: Optional[int] = None,
    ) -> None:
        """标记文件处理失败。"""


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """基于 SQLite 的默认完整性检查实现。

    设计说明：
    1. 数据库默认路径：`data/db/ingestion_history.db`。
    2. 启用 WAL 模式，提高并发写入能力。
    3. 通过 `file_hash + status=success` 判定增量跳过。
    """

    def __init__(self, db_path: str = "data/db/ingestion_history.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = Lock()
        self._init_db()

    def compute_sha256(self, path: str) -> str:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"file not found: {file_path}")

        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        if not file_hash:
            return False

        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM ingestion_history WHERE file_hash = ? AND status = 'success' LIMIT 1",
                (file_hash,),
            ).fetchone()
        return row is not None

    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        file_size: Optional[int] = None,
        chunk_count: Optional[int] = None,
    ) -> None:
        self._upsert_status(
            file_hash=file_hash,
            status="success",
            file_path=file_path,
            file_size=file_size,
            error_msg=None,
            chunk_count=chunk_count,
        )

    def mark_failed(
        self,
        file_hash: str,
        error_msg: str,
        file_path: str = "",
        file_size: Optional[int] = None,
    ) -> None:
        self._upsert_status(
            file_hash=file_hash,
            status="failed",
            file_path=file_path,
            file_size=file_size,
            error_msg=error_msg,
            chunk_count=None,
        )

    def remove_record(
        self,
        file_hash: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> int:
        """按 file_hash 或 file_path 删除完整性记录。"""
        if not file_hash and not file_path:
            raise ValueError("file_hash or file_path is required")

        with self._write_lock:
            with self._connect() as conn:
                if file_hash:
                    cursor = conn.execute(
                        "DELETE FROM ingestion_history WHERE file_hash = ?",
                        (file_hash,),
                    )
                else:
                    cursor = conn.execute(
                        "DELETE FROM ingestion_history WHERE file_path = ?",
                        (file_path,),
                    )
                conn.commit()
        return cursor.rowcount

    def list_processed(self) -> List[Dict[str, object]]:
        """列出完整性记录，供 Dashboard/DocumentManager 使用。"""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT file_hash, file_path, file_size, status, processed_at, error_msg, chunk_count
                FROM ingestion_history
                ORDER BY processed_at DESC
                """
            ).fetchall()
        return [
            {
                "file_hash": row[0],
                "file_path": row[1],
                "file_size": row[2],
                "status": row[3],
                "processed_at": row[4],
                "error_msg": row[5],
                "chunk_count": row[6],
            }
            for row in rows
        ]

    def _upsert_status(
        self,
        file_hash: str,
        status: str,
        file_path: str,
        file_size: Optional[int],
        error_msg: Optional[str],
        chunk_count: Optional[int],
    ) -> None:
        if not file_hash:
            raise ValueError("file_hash is required")

        with self._write_lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO ingestion_history (
                        file_hash, file_path, file_size, status, error_msg, chunk_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(file_hash) DO UPDATE SET
                        file_path=excluded.file_path,
                        file_size=excluded.file_size,
                        status=excluded.status,
                        error_msg=excluded.error_msg,
                        chunk_count=excluded.chunk_count,
                        processed_at=CURRENT_TIMESTAMP
                    """,
                    (file_hash, file_path, file_size, status, error_msg, chunk_count),
                )
                conn.commit()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'processing')),
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_msg TEXT,
                    chunk_count INTEGER
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON ingestion_history(status)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_processed_at ON ingestion_history(processed_at)"
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False + WAL 可提升并发访问兼容性。
        return sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
