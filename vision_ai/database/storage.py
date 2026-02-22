from __future__ import annotations

import csv
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet


@dataclass
class EmployeeEmbedding:
    employee_id: str
    name: str
    embedding: np.ndarray
    created_at: str
    updated_at: str


class EmbeddingCipher:
    def __init__(self, key_path: Path):
        self.key_path = key_path
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        key = self._load_or_create_key()
        self.fernet = Fernet(key)

    def _load_or_create_key(self) -> bytes:
        env_key = os.getenv("VISION_AI_EMBED_KEY")
        if env_key:
            return env_key.encode("utf-8")
        if self.key_path.exists():
            return self.key_path.read_bytes().strip()

        key = Fernet.generate_key()
        self.key_path.write_bytes(key)
        return key

    def encrypt(self, embedding: np.ndarray) -> bytes:
        raw = np.asarray(embedding, dtype=np.float32).tobytes()
        return self.fernet.encrypt(raw)

    def decrypt(self, blob: bytes, dim: int) -> np.ndarray:
        raw = self.fernet.decrypt(blob)
        vector = np.frombuffer(raw, dtype=np.float32, count=dim)
        return vector.copy()


class VisionAIDatabase:
    def __init__(self, db_path: Path, key_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cipher = EmbeddingCipher(key_path=key_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS employees (
                    employee_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    embedding_encrypted BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT NOT NULL,
                    check_in_time TEXT NOT NULL,
                    attendance_date TEXT NOT NULL,
                    FOREIGN KEY (employee_id) REFERENCES employees(employee_id),
                    UNIQUE(employee_id, attendance_date)
                );

                CREATE TABLE IF NOT EXISTS ocr_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    captured_at TEXT NOT NULL,
                    text_payload TEXT NOT NULL,
                    source TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS roles (
                    role_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role_name TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (role_id) REFERENCES roles(role_id)
                );
                """
            )
            conn.execute("INSERT OR IGNORE INTO roles (role_name) VALUES ('admin')")
            conn.execute("INSERT OR IGNORE INTO roles (role_name) VALUES ('operator')")
            conn.execute("INSERT OR IGNORE INTO roles (role_name) VALUES ('viewer')")

    def upsert_employee_embedding(self, employee_id: str, name: str, embedding: np.ndarray) -> None:
        embedding = np.asarray(embedding, dtype=np.float32)
        now = datetime.now().isoformat(timespec="seconds")
        encrypted = self.cipher.encrypt(embedding)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO employees (
                    employee_id, name, embedding_encrypted, embedding_dim, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(employee_id) DO UPDATE SET
                    name = excluded.name,
                    embedding_encrypted = excluded.embedding_encrypted,
                    embedding_dim = excluded.embedding_dim,
                    updated_at = excluded.updated_at
                """,
                (employee_id, name, encrypted, int(embedding.size), now, now),
            )

    def load_employee_embeddings(self) -> list[EmployeeEmbedding]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT employee_id, name, embedding_encrypted, embedding_dim, created_at, updated_at
                FROM employees
                ORDER BY created_at ASC
                """
            ).fetchall()

        output: list[EmployeeEmbedding] = []
        for row in rows:
            vector = self.cipher.decrypt(row["embedding_encrypted"], int(row["embedding_dim"]))
            output.append(
                EmployeeEmbedding(
                    employee_id=row["employee_id"],
                    name=row["name"],
                    embedding=vector,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return output

    def mark_attendance(self, employee_id: str, at_time: datetime) -> bool:
        at = at_time.isoformat(timespec="seconds")
        day = at_time.date().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO attendance (employee_id, check_in_time, attendance_date)
                VALUES (?, ?, ?)
                """,
                (employee_id, at, day),
            )
            return cursor.rowcount == 1

    def today_marked_employee_ids(self) -> set[str]:
        today = datetime.now().date().isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT employee_id FROM attendance WHERE attendance_date = ?",
                (today,),
            ).fetchall()
        return {row["employee_id"] for row in rows}

    def daily_attendance_summary(self, day: str | None = None) -> list[dict[str, str]]:
        day = day or datetime.now().date().isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT a.employee_id, e.name, a.check_in_time, a.attendance_date
                FROM attendance a
                JOIN employees e ON e.employee_id = a.employee_id
                WHERE a.attendance_date = ?
                ORDER BY a.check_in_time ASC
                """,
                (day,),
            ).fetchall()
        return [
            {
                "employee_id": row["employee_id"],
                "name": row["name"],
                "check_in_time": row["check_in_time"],
                "attendance_date": row["attendance_date"],
            }
            for row in rows
        ]

    def export_attendance_csv(self, output_path: Path, day: str | None = None) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.daily_attendance_summary(day=day)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["employee_id", "name", "check_in_time", "attendance_date"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return output_path

    def save_ocr_snapshot(self, lines: list[str], source: str = "live_camera") -> None:
        payload = "\n".join(lines) if lines else "(empty)"
        captured_at = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ocr_snapshots (captured_at, text_payload, source)
                VALUES (?, ?, ?)
                """,
                (captured_at, payload, source),
            )
