from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RecognitionMatchRequest(BaseModel):
    device_id: uuid.UUID
    embedding: list[float] = Field(min_length=512, max_length=512)
    timestamp: datetime | None = None
    location: str | None = None
    metadata: dict[str, Any] = {}


class RecognitionEventPayload(BaseModel):
    event_id: uuid.UUID
    employee_id: uuid.UUID | None
    employee_name: str | None
    confidence: float
    matched: bool
    device_id: uuid.UUID
    timestamp: datetime
    location: str | None = None


class RecognitionMatchResponse(BaseModel):
    matched: bool
    confidence: float
    employee_id: uuid.UUID | None
    employee_name: str | None
    event_id: uuid.UUID
    attendance_logged: bool
    source: str = "local_backend"

