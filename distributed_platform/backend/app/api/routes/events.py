from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.api.deps import db_session, require_roles
from app.db.models import RecognitionEvent
from app.schemas.auth import CurrentPrincipal
from app.schemas.event import RecognitionEventResponse

router = APIRouter(prefix="/events", tags=["events"])


@router.get("/recognitions", response_model=list[RecognitionEventResponse])
def list_recognition_events(
    limit: int = 200,
    _principal: CurrentPrincipal = Depends(require_roles("admin", "manager", "operator")),
    db: Session = db_session(),
):
    rows = db.scalars(
        select(RecognitionEvent)
        .order_by(desc(RecognitionEvent.timestamp))
        .limit(max(1, min(1000, limit)))
    ).all()
    return rows

