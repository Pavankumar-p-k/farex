from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class AttendanceResponse(BaseModel):
    id: uuid.UUID
    employee_id: uuid.UUID
    device_id: uuid.UUID
    timestamp: datetime
    location: str | None = None

    class Config:
        from_attributes = True

