from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.api.deps import db_session, require_roles
from app.db.models import Attendance
from app.schemas.attendance import AttendanceResponse
from app.schemas.auth import CurrentPrincipal

router = APIRouter(prefix="/attendance", tags=["attendance"])


@router.get("", response_model=list[AttendanceResponse])
def list_attendance(
    limit: int = 200,
    _principal: CurrentPrincipal = Depends(require_roles("admin", "manager", "operator")),
    db: Session = db_session(),
):
    rows = db.scalars(select(Attendance).order_by(desc(Attendance.timestamp)).limit(max(1, min(1000, limit)))).all()
    return rows

