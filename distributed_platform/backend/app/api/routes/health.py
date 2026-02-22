from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "service": "distributed-face-platform",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

