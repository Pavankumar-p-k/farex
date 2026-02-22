from __future__ import annotations

from typing import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.security import safe_decode_token
from app.db.session import get_db
from app.schemas.auth import CurrentPrincipal

bearer_scheme = HTTPBearer(auto_error=False)


def db_session() -> Session:
    return Depends(get_db)  # type: ignore[return-value]


def get_current_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> CurrentPrincipal:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token.")
    payload = safe_decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token.")
    return CurrentPrincipal(
        subject=str(payload.get("sub", "")),
        role=str(payload.get("role", "")),
        device_id=payload.get("device_id"),
    )


def require_roles(*allowed_roles: str) -> Callable[[CurrentPrincipal], CurrentPrincipal]:
    allowed = set(allowed_roles)

    def _checker(principal: CurrentPrincipal = Depends(get_current_principal)) -> CurrentPrincipal:
        if principal.role not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role.")
        return principal

    return _checker

