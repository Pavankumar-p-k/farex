import os
from io import BytesIO
from typing import Any

import pandas as pd
from werkzeug.security import check_password_hash, generate_password_hash

from .database import AttendanceDatabase
from .exceptions import AttendanceError


class AdminService:
    def __init__(self, db: AttendanceDatabase):
        self.db = db

    def ensure_default_admin(self) -> str:
        username = os.getenv("ADMIN_USERNAME", "admin").strip().lower()
        password = os.getenv("ADMIN_PASSWORD", "admin123").strip()
        existing = self.db.get_admin_user(username)
        if existing is None:
            self.db.upsert_admin_user(username=username, password_hash=generate_password_hash(password))
        return username

    def verify_login(self, username: str, password: str) -> bool:
        user = self.db.get_admin_user(username)
        if user is None:
            return False
        return check_password_hash(user["password_hash"], password)

    def stats(self) -> dict[str, int]:
        return self.db.attendance_stats()

    def attendance_rows(self, query_text: str, date_from: str, date_to: str, limit: int = 2000):
        return self.db.search_attendance(
            query_text=query_text,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )

    def attendance_excel(self, query_text: str, date_from: str, date_to: str) -> bytes:
        rows = self.attendance_rows(query_text=query_text, date_from=date_from, date_to=date_to, limit=10_000)
        data: list[dict[str, Any]] = [
            {
                "Record ID": row.id,
                "Employee ID": row.employee_id,
                "Name": row.name,
                "Check In Time": row.check_in_time,
                "Attendance Date": row.attendance_date,
            }
            for row in rows
        ]

        df = pd.DataFrame(data)
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Attendance")
            ws = writer.sheets["Attendance"]
            ws.freeze_panes = "A2"

        output.seek(0)
        return output.read()

    def delete_employee(self, employee_id: str) -> None:
        removed = self.db.delete_employee(employee_id.strip())
        if not removed:
            raise AttendanceError(f"Employee {employee_id} not found.")

