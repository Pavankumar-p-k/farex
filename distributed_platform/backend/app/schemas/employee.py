from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class EmployeeCreate(BaseModel):
    external_id: str = Field(min_length=2, max_length=64)
    name: str = Field(min_length=2, max_length=120)
    role: str = "employee"
    embedding: list[float] = Field(min_length=512, max_length=512)


class EmployeeResponse(BaseModel):
    id: uuid.UUID
    external_id: str
    name: str
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

