from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.api.deps import db_session, require_roles
from app.core.config import get_settings
from app.core.security import normalize_embedding
from app.db.models import Attendance, Device, RecognitionEvent
from app.schemas.auth import CurrentPrincipal
from app.schemas.recognition import (
    RecognitionEventPayload,
    RecognitionMatchRequest,
    RecognitionMatchResponse,
)
from app.services.matcher import get_matcher
from app.ws.manager import ws_manager

router = APIRouter(prefix="/recognitions", tags=["recognitions"])
settings = get_settings()
matcher = get_matcher()


def _resolve_device_or_404(db: Session, device_id):
    row = db.scalar(select(Device).where(Device.id == device_id))
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown device '{device_id}'. Send /devices/heartbeat first.",
        )
    row.last_seen_at = datetime.now(timezone.utc)
    row.status = "online"
    return row


@router.post("/match", response_model=RecognitionMatchResponse)
async def match_embedding(
    payload: RecognitionMatchRequest,
    _principal: CurrentPrincipal = Depends(require_roles("admin", "manager", "operator", "device")),
    db: Session = db_session(),
):
    normalized = normalize_embedding(payload.embedding)
    _resolve_device_or_404(db, payload.device_id)
    result = matcher.match(db, normalized)

    event = RecognitionEvent(
        employee_id=result.employee_id,
        device_id=payload.device_id,
        confidence=result.confidence,
        matched=result.matched,
        payload_json=payload.metadata,
        timestamp=payload.timestamp or datetime.now(timezone.utc),
    )
    db.add(event)

    attendance_logged = False
    if result.matched and result.employee_id is not None:
        window_start = (payload.timestamp or datetime.now(timezone.utc)) - timedelta(
            seconds=settings.attendance_cooldown_seconds
        )
        recent = db.scalar(
            select(Attendance)
            .where(
                Attendance.employee_id == result.employee_id,
                Attendance.timestamp >= window_start,
            )
            .order_by(desc(Attendance.timestamp))
        )
        if recent is None:
            attendance = Attendance(
                employee_id=result.employee_id,
                device_id=payload.device_id,
                timestamp=payload.timestamp or datetime.now(timezone.utc),
                location=payload.location,
            )
            db.add(attendance)
            attendance_logged = True

    db.commit()
    db.refresh(event)

    ws_payload = RecognitionEventPayload(
        event_id=event.id,
        employee_id=result.employee_id,
        employee_name=result.employee_name,
        confidence=result.confidence,
        matched=result.matched,
        device_id=payload.device_id,
        timestamp=event.timestamp,
        location=payload.location,
    )
    await ws_manager.broadcast({"type": "recognition_event", "payload": ws_payload.model_dump(mode="json")})

    return RecognitionMatchResponse(
        matched=result.matched,
        confidence=result.confidence,
        employee_id=result.employee_id,
        employee_name=result.employee_name,
        event_id=event.id,
        attendance_logged=attendance_logged,
    )
