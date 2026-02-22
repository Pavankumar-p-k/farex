from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class DeviceHeartbeat(BaseModel):
    device_id: uuid.UUID | None = None
    device_name: str
    device_type: str
    network_mode: str = "local"
    status: str = "online"
    metadata: dict[str, Any] = {}


class DeviceResponse(BaseModel):
    id: uuid.UUID
    device_name: str
    device_type: str
    status: str
    network_mode: str
    last_seen_at: datetime

    class Config:
        from_attributes = True

