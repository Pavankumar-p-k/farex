from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class QueuedEmbedding:
    id: int
    payload: dict[str, Any]


class OfflineEmbeddingQueue:
    def __init__(self, db_path: str, max_items: int = 5000) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_items = max(100, int(max_items))
        self._lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists embedding_queue (
                    id integer primary key autoincrement,
                    payload_json text not null,
                    created_at text not null
                )
                """
            )
            conn.commit()

    def enqueue(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"))
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "insert into embedding_queue (payload_json, created_at) values (?, ?)",
                (body, now),
            )
            row = conn.execute("select count(*) as c from embedding_queue").fetchone()
            count = int(row["c"]) if row else 0
            overflow = max(0, count - self.max_items)
            if overflow > 0:
                conn.execute(
                    """
                    delete from embedding_queue
                    where id in (
                        select id from embedding_queue order by id asc limit ?
                    )
                    """,
                    (overflow,),
                )
            conn.commit()

    def fetch_batch(self, limit: int) -> list[QueuedEmbedding]:
        batch_size = max(1, int(limit))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "select id, payload_json from embedding_queue order by id asc limit ?",
                (batch_size,),
            ).fetchall()
        items: list[QueuedEmbedding] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                payload = {}
            items.append(QueuedEmbedding(id=int(row["id"]), payload=payload))
        return items

    def acknowledge(self, ids: list[int]) -> None:
        if not ids:
            return
        clean_ids = [int(item) for item in ids]
        placeholders = ",".join("?" for _ in clean_ids)
        with self._lock, self._connect() as conn:
            conn.execute(f"delete from embedding_queue where id in ({placeholders})", clean_ids)
            conn.commit()

    def size(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("select count(*) as c from embedding_queue").fetchone()
        return int(row["c"]) if row else 0

