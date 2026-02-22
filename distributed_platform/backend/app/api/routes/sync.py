from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import require_roles
from app.schemas.auth import CurrentPrincipal
from app.services.sync import SupabaseSyncWorker

router = APIRouter(prefix="/sync", tags=["sync"])
sync_worker = SupabaseSyncWorker()


@router.post("/trigger")
def trigger_sync(_principal: CurrentPrincipal = Depends(require_roles("admin", "manager"))):
    sync_worker.sync_once()
    return {"ok": True, "message": "Manual sync completed."}


@router.get("/status")
def sync_status(_principal: CurrentPrincipal = Depends(require_roles("admin", "manager", "operator"))):
    return {"enabled": sync_worker.enabled}

