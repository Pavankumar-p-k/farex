from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class RecognitionEventResponse(BaseModel):
    id: uuid.UUID
    employee_id: uuid.UUID | None
    device_id: uuid.UUID
    confidence: float
    matched: bool
    timestamp: datetime

    class Config:
        from_attributes = True

