from datetime import datetime

from .database import AttendanceDatabase


class AttendanceService:
    def __init__(self, db: AttendanceDatabase):
        self.db = db

    def mark_now(self, employee_id: str) -> bool:
        return self.db.mark_attendance(employee_id=employee_id, check_in_time=datetime.now())

