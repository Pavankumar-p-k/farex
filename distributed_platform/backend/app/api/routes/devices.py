from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import db_session, require_roles
from app.db.models import Device
from app.schemas.auth import CurrentPrincipal
from app.schemas.device import DeviceHeartbeat, DeviceResponse

router = APIRouter(prefix="/devices", tags=["devices"])


@router.post("/heartbeat", response_model=DeviceResponse)
def heartbeat(
    payload: DeviceHeartbeat,
    _principal: CurrentPrincipal = Depends(require_roles("admin", "manager", "operator", "device")),
    db: Session = db_session(),
):
    row: Device | None = None
    if payload.device_id is not None:
        row = db.scalar(select(Device).where(Device.id == payload.device_id))

    if row is None:
        row = Device(
            id=payload.device_id or uuid.uuid4(),
            device_name=payload.device_name,
            device_type=payload.device_type,
            network_mode=payload.network_mode,
            status=payload.status,
            metadata_json=payload.metadata,
            last_seen_at=datetime.now(timezone.utc),
        )
        db.add(row)
    else:
        row.device_name = payload.device_name
        row.device_type = payload.device_type
        row.network_mode = payload.network_mode
        row.status = payload.status
        row.metadata_json = payload.metadata
        row.last_seen_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(row)
    return row

