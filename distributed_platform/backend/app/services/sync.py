from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

import requests
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import Attendance, Employee, RecognitionEvent
from app.db.session import SessionLocal

logger = logging.getLogger("distributed.sync")


class SupabaseSyncWorker:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def enabled(self) -> bool:
        return (
            self.settings.sync_enabled
            and bool(self.settings.supabase_url.strip())
            and bool(self.settings.supabase_service_key.strip())
        )

    def start(self) -> None:
        if not self.enabled:
            logger.info("Supabase sync disabled (missing credentials or disabled in env).")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="supabase-sync", daemon=True)
        self._thread.start()
        logger.info("Supabase sync worker started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.sync_once()
            except Exception:
                logger.exception("Supabase sync iteration failed")
            self._stop_event.wait(max(5, self.settings.sync_interval_seconds))

    def sync_once(self) -> None:
        headers = {
            "apikey": self.settings.supabase_service_key,
            "Authorization": f"Bearer {self.settings.supabase_service_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        }
        base = f"{self.settings.supabase_url.rstrip('/')}/rest/v1"
        now = datetime.now(timezone.utc)
        with SessionLocal() as db:
            employee_rows = db.scalars(select(Employee).where(Employee.synced_at.is_(None))).all()
            attendance_rows = db.scalars(select(Attendance).where(Attendance.synced_at.is_(None))).all()
            event_rows = db.scalars(select(RecognitionEvent).where(RecognitionEvent.synced_at.is_(None))).all()

            if employee_rows:
                payload = [
                    {
                        "id": str(row.id),
                        "external_id": row.external_id,
                        "name": row.name,
                        "role": row.role,
                        "embedding_ciphertext": row.embedding_ciphertext,
                        "embedding_norm": row.embedding_norm,
                        "is_active": row.is_active,
                    }
                    for row in employee_rows
                ]
                requests.post(f"{base}/employees", json=payload, headers=headers, timeout=15).raise_for_status()
                for row in employee_rows:
                    row.synced_at = now

            if attendance_rows:
                payload = [
                    {
                        "id": str(row.id),
                        "employee_id": str(row.employee_id),
                        "device_id": str(row.device_id),
                        "timestamp": row.timestamp.isoformat(),
                        "location": row.location,
                    }
                    for row in attendance_rows
                ]
                requests.post(f"{base}/attendance", json=payload, headers=headers, timeout=15).raise_for_status()
                for row in attendance_rows:
                    row.synced_at = now

            if event_rows:
                payload = [
                    {
                        "id": str(row.id),
                        "employee_id": str(row.employee_id) if row.employee_id else None,
                        "device_id": str(row.device_id),
                        "confidence": row.confidence,
                        "matched": row.matched,
                        "timestamp": row.timestamp.isoformat(),
                        "payload_json": row.payload_json,
                    }
                    for row in event_rows
                ]
                requests.post(
                    f"{base}/recognition_events",
                    json=payload,
                    headers=headers,
                    timeout=15,
                ).raise_for_status()
                for row in event_rows:
                    row.synced_at = now

            db.commit()

