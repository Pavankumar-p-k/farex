from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import db_session, require_roles
from app.core.security import normalize_embedding
from app.db.models import Employee
from app.schemas.auth import CurrentPrincipal
from app.schemas.employee import EmployeeCreate, EmployeeResponse
from app.services.encryption import embedding_crypto
from app.services.matcher import get_matcher

router = APIRouter(prefix="/employees", tags=["employees"])
matcher_singleton = get_matcher()


@router.get("", response_model=list[EmployeeResponse])
def list_employees(
    _principal: CurrentPrincipal = Depends(require_roles("admin", "manager", "operator")),
    db: Session = db_session(),
):
    rows = db.scalars(select(Employee).where(Employee.is_active.is_(True)).order_by(Employee.created_at.desc())).all()
    return rows


@router.post("", response_model=EmployeeResponse)
def create_employee(
    payload: EmployeeCreate,
    _principal: CurrentPrincipal = Depends(require_roles("admin", "manager")),
    db: Session = db_session(),
):
    embedding = normalize_embedding(payload.embedding)
    row = Employee(
        external_id=payload.external_id,
        name=payload.name,
        role=payload.role,
        embedding_ciphertext=embedding_crypto.encrypt(embedding),
        embedding_norm=1.0,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    matcher_singleton.invalidate()
    return row
