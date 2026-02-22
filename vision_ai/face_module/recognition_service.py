from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock

import numpy as np

from vision_ai.database import VisionAIDatabase
from vision_ai.face_module.arcface_engine import ArcFaceDetection, ArcFaceEngine
from vision_ai.utils.types import FaceIdentity, FaceResult


@dataclass
class RegistrationSession:
    employee_id: str
    name: str
    target_samples: int
    embeddings: list[np.ndarray] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)

    @property
    def progress(self) -> float:
        if self.target_samples <= 0:
            return 0.0
        return min(1.0, len(self.embeddings) / float(self.target_samples))

    @property
    def completed(self) -> bool:
        return len(self.embeddings) >= self.target_samples


class FaceRecognitionService:
    def __init__(
        self,
        engine: ArcFaceEngine,
        db: VisionAIDatabase,
        match_threshold: float = 0.55,
    ) -> None:
        self.engine = engine
        self.db = db
        self.match_threshold = match_threshold
        self._lock = Lock()
        self._session: RegistrationSession | None = None
        self._session_message = ""
        self._known_ids: list[str] = []
        self._known_names: list[str] = []
        self._known_embeddings = np.empty((0, 512), dtype=np.float32)
        self._today_marked = self.db.today_marked_employee_ids()
        self.reload_gallery()

    def start_registration(self, employee_id: str, name: str, samples: int) -> str:
        employee_id = employee_id.strip()
        name = name.strip()
        if not employee_id or not name:
            return "Registration failed: employee_id and name are required."

        target = max(20, min(30, int(samples)))
        with self._lock:
            self._session = RegistrationSession(
                employee_id=employee_id,
                name=name,
                target_samples=target,
            )
            self._session_message = f"Registration started for {employee_id} ({name}) [{target} samples]"
            return self._session_message

    def reload_gallery(self) -> None:
        employees = self.db.load_employee_embeddings()
        if not employees:
            self._known_ids = []
            self._known_names = []
            self._known_embeddings = np.empty((0, 512), dtype=np.float32)
            return

        self._known_ids = [employee.employee_id for employee in employees]
        self._known_names = [employee.name for employee in employees]
        matrix = np.stack([employee.embedding for employee in employees], axis=0).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        self._known_embeddings = matrix / norms

    def process_frame(self, frame_id: int, frame_bgr: np.ndarray) -> FaceResult:
        detections = self.engine.detect(frame_bgr)
        with self._lock:
            session_active = self._session is not None
        if session_active:
            self._collect_registration_sample(detections)

        identities: list[FaceIdentity] = []
        for detection in detections:
            identity = self._recognize_detection(detection)
            identities.append(identity)

        with self._lock:
            session = self._session
            message = self._session_message
        return FaceResult(
            frame_id=frame_id,
            identities=identities,
            registration_mode=session is not None,
            registration_progress=session.progress if session else 0.0,
            message=message,
        )

    def _collect_registration_sample(self, detections: list[ArcFaceDetection]) -> None:
        if not detections:
            return
        with self._lock:
            session = self._session
        if session is None:
            return

        primary = detections[0]
        session.embeddings.append(primary.embedding)

        if session.completed:
            stacked = np.stack(session.embeddings, axis=0)
            mean_embedding = np.mean(stacked, axis=0).astype(np.float32)
            norm = np.linalg.norm(mean_embedding)
            if norm > 1e-8:
                mean_embedding /= norm

            self.db.upsert_employee_embedding(
                employee_id=session.employee_id,
                name=session.name,
                embedding=mean_embedding,
            )
            self.reload_gallery()
            with self._lock:
                self._session_message = (
                    f"Registration completed: {session.employee_id} ({session.name}), "
                    f"samples={len(session.embeddings)}"
                )
                self._session = None
        else:
            with self._lock:
                self._session_message = (
                    f"Collecting registration: {session.employee_id} ({session.name}) "
                    f"{len(session.embeddings)}/{session.target_samples}"
                )

    def _recognize_detection(self, detection: ArcFaceDetection) -> FaceIdentity:
        if self._known_embeddings.size == 0:
            return FaceIdentity(
                employee_id="UNKNOWN",
                name="Unknown",
                confidence=0.0,
                bbox=detection.bbox,
                new_attendance=False,
            )

        similarities = self._known_embeddings @ detection.embedding
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score < self.match_threshold:
            return FaceIdentity(
                employee_id="UNKNOWN",
                name="Unknown",
                confidence=best_score,
                bbox=detection.bbox,
                new_attendance=False,
            )

        employee_id = self._known_ids[best_idx]
        name = self._known_names[best_idx]

        new_attendance = False
        if employee_id not in self._today_marked:
            if self.db.mark_attendance(employee_id=employee_id, at_time=datetime.now()):
                self._today_marked.add(employee_id)
                new_attendance = True

        return FaceIdentity(
            employee_id=employee_id,
            name=name,
            confidence=best_score,
            bbox=detection.bbox,
            new_attendance=new_attendance,
        )
