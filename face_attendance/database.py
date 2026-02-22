import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .exceptions import DatabaseError


@dataclass
class EmployeeRecord:
    employee_id: str
    name: str
    encoding: np.ndarray


@dataclass
class EmployeeProfile:
    employee_id: str
    name: str
    created_at: str
    updated_at: str


@dataclass
class AttendanceRecord:
    id: int
    employee_id: str
    name: str
    check_in_time: str
    attendance_date: str


class AttendanceDatabase:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _initialize(self) -> None:
        try:
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS employees (
                        employee_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        face_encoding BLOB NOT NULL,
                        encoding_dim INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        employee_id TEXT NOT NULL,
                        check_in_time TEXT NOT NULL,
                        attendance_date TEXT NOT NULL,
                        FOREIGN KEY (employee_id) REFERENCES employees(employee_id),
                        -- Ensures one attendance row per employee per day.
                        UNIQUE(employee_id, attendance_date)
                    );

                    CREATE TABLE IF NOT EXISTS admin_users (
                        username TEXT PRIMARY KEY,
                        password_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to initialize database: {exc}") from exc

    def upsert_employee(self, employee_id: str, name: str, encoding: np.ndarray) -> None:
        if encoding.ndim != 1:
            raise DatabaseError("Encoding must be a 1D vector.")

        now = datetime.now().isoformat(timespec="seconds")
        vector = np.asarray(encoding, dtype=np.float32)
        blob = vector.tobytes()

        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO employees (
                        employee_id, name, face_encoding, encoding_dim, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(employee_id) DO UPDATE SET
                        name = excluded.name,
                        face_encoding = excluded.face_encoding,
                        encoding_dim = excluded.encoding_dim,
                        updated_at = excluded.updated_at
                    """,
                    (employee_id, name, blob, vector.size, now, now),
                )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to save employee {employee_id}: {exc}") from exc

    def list_employees(self) -> List[EmployeeRecord]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT employee_id, name, face_encoding, encoding_dim
                    FROM employees
                    ORDER BY employee_id ASC
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to load employees: {exc}") from exc

        records: List[EmployeeRecord] = []
        for row in rows:
            encoding = np.frombuffer(row["face_encoding"], dtype=np.float32, count=row["encoding_dim"])
            records.append(
                EmployeeRecord(
                    employee_id=row["employee_id"],
                    name=row["name"],
                    encoding=encoding.copy(),
                )
            )
        return records

    def list_employee_profiles(self) -> List[EmployeeProfile]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT employee_id, name, created_at, updated_at
                    FROM employees
                    ORDER BY created_at DESC
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to load employee profiles: {exc}") from exc

        return [
            EmployeeProfile(
                employee_id=row["employee_id"],
                name=row["name"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def delete_employee(self, employee_id: str) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM attendance WHERE employee_id = ?", (employee_id,))
                cursor = conn.execute("DELETE FROM employees WHERE employee_id = ?", (employee_id,))
                return cursor.rowcount > 0
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to delete employee {employee_id}: {exc}") from exc

    def mark_attendance(self, employee_id: str, check_in_time: datetime) -> bool:
        attendance_date = check_in_time.date().isoformat()
        timestamp = check_in_time.isoformat(timespec="seconds")

        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO attendance (employee_id, check_in_time, attendance_date)
                    VALUES (?, ?, ?)
                    """,
                    (employee_id, timestamp, attendance_date),
                )
                return cursor.rowcount == 1
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to mark attendance for {employee_id}: {exc}") from exc

    def today_marked_employee_ids(self) -> set[str]:
        today = datetime.now().date().isoformat()
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT employee_id FROM attendance WHERE attendance_date = ?",
                    (today,),
                ).fetchall()
                return {row["employee_id"] for row in rows}
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to query today's attendance: {exc}") from exc

    def search_attendance(
        self,
        query_text: str = "",
        date_from: str = "",
        date_to: str = "",
        limit: int = 2000,
    ) -> List[AttendanceRecord]:
        sql = """
            SELECT a.id, a.employee_id, e.name, a.check_in_time, a.attendance_date
            FROM attendance a
            JOIN employees e ON e.employee_id = a.employee_id
            WHERE 1=1
        """
        params: List[Any] = []

        if query_text.strip():
            term = f"%{query_text.strip()}%"
            sql += " AND (a.employee_id LIKE ? OR e.name LIKE ?)"
            params.extend([term, term])
        if date_from.strip():
            sql += " AND a.attendance_date >= ?"
            params.append(date_from.strip())
        if date_to.strip():
            sql += " AND a.attendance_date <= ?"
            params.append(date_to.strip())

        safe_limit = max(1, min(10_000, int(limit)))
        sql += " ORDER BY a.check_in_time DESC LIMIT ?"
        params.append(safe_limit)

        try:
            with self._connect() as conn:
                rows = conn.execute(sql, tuple(params)).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to search attendance: {exc}") from exc

        return [
            AttendanceRecord(
                id=row["id"],
                employee_id=row["employee_id"],
                name=row["name"],
                check_in_time=row["check_in_time"],
                attendance_date=row["attendance_date"],
            )
            for row in rows
        ]

    def attendance_stats(self) -> dict[str, int]:
        today = datetime.now().date().isoformat()
        try:
            with self._connect() as conn:
                employees = conn.execute("SELECT COUNT(*) AS c FROM employees").fetchone()["c"]
                total = conn.execute("SELECT COUNT(*) AS c FROM attendance").fetchone()["c"]
                today_count = conn.execute(
                    "SELECT COUNT(*) AS c FROM attendance WHERE attendance_date = ?",
                    (today,),
                ).fetchone()["c"]
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to load attendance stats: {exc}") from exc

        return {
            "employees": int(employees),
            "attendance_total": int(total),
            "attendance_today": int(today_count),
        }

    def upsert_admin_user(self, username: str, password_hash: str) -> None:
        username = username.strip().lower()
        now = datetime.now().isoformat(timespec="seconds")
        if not username:
            raise DatabaseError("Admin username cannot be empty.")
        if not password_hash.strip():
            raise DatabaseError("Admin password hash cannot be empty.")

        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO admin_users (username, password_hash, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(username) DO UPDATE SET
                        password_hash = excluded.password_hash,
                        updated_at = excluded.updated_at
                    """,
                    (username, password_hash, now, now),
                )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to save admin user {username}: {exc}") from exc

    def get_admin_user(self, username: str) -> Optional[dict[str, str]]:
        username = username.strip().lower()
        if not username:
            return None

        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT username, password_hash, created_at, updated_at
                    FROM admin_users
                    WHERE username = ?
                    """,
                    (username,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to load admin user {username}: {exc}") from exc

        if row is None:
            return None
        return {
            "username": row["username"],
            "password_hash": row["password_hash"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
