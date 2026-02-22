import time
from datetime import date
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .attendance_service import AttendanceService
from .camera import CameraStream
from .config import (
    ATTENDANCE_COOLDOWN_SECONDS,
    RECOGNITION_MARGIN,
    RECOGNITION_THRESHOLD,
    SINGLE_USER_RECOGNITION_THRESHOLD,
)
from .database import AttendanceDatabase, EmployeeRecord
from .exceptions import AttendanceError
from .face_engine import FaceEngine
from .logger import setup_logger


class RecognitionService:
    def __init__(self, db: AttendanceDatabase, engine: FaceEngine, threshold: float = RECOGNITION_THRESHOLD):
        self.db = db
        self.engine = engine
        self.threshold = threshold
        self.attendance = AttendanceService(db)
        self.logger = setup_logger(self.__class__.__name__)

        self.employee_ids: List[str] = []
        self.employee_names: List[str] = []
        self.known_matrix: Optional[np.ndarray] = None

        self.current_day = date.today().isoformat()
        self.marked_today = self.db.today_marked_employee_ids()
        self.last_attempt_times: Dict[str, float] = {}

        self._refresh_known_faces()

    def run(self, camera_index: int = 0) -> None:
        if self.known_matrix is None or not len(self.employee_ids):
            raise AttendanceError("No registered employees found. Run registration first.")

        window_name = "Live Recognition - Press Q to exit"
        self.logger.info("Starting real-time recognition loop")

        with CameraStream(camera_index) as cam:
            while True:
                frame = cam.read()
                self._rollover_day_if_needed()
                status_events: List[str] = []

                batch = self.engine.extract_embeddings(frame)
                for emb, box in zip(batch.embeddings, batch.boxes):
                    employee_id, name, score = self._match_face(emb)
                    event = self._handle_attendance_event(employee_id, name)
                    if event:
                        status_events.append(event)
                    self._draw_face_box(frame, box, name, score, employee_id is not None)

                self._draw_status_bar(frame, status_events)
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    def _refresh_known_faces(self) -> None:
        records: List[EmployeeRecord] = self.db.list_employees()
        if not records:
            self.known_matrix = None
            self.employee_ids = []
            self.employee_names = []
            return

        self.employee_ids = [rec.employee_id for rec in records]
        self.employee_names = [rec.name for rec in records]

        matrix = np.vstack([rec.encoding for rec in records]).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        self.known_matrix = matrix / norms

    def _match_face(self, query: np.ndarray) -> Tuple[Optional[str], str, float]:
        if self.known_matrix is None or self.known_matrix.size == 0:
            return None, "Unknown", 0.0

        query = query.astype(np.float32)
        norm = np.linalg.norm(query)
        if norm == 0.0:
            return None, "Unknown", 0.0
        query = query / norm

        scores = self.known_matrix @ query
        idx = int(np.argmax(scores))
        best = float(scores[idx])

        if scores.size == 1:
            required = max(self.threshold + 0.08, SINGLE_USER_RECOGNITION_THRESHOLD)
            if best >= required:
                return self.employee_ids[idx], self.employee_names[idx], best
            return None, "Unknown", best

        second = float(np.partition(scores, -2)[-2])
        margin = best - second
        if best >= self.threshold and margin >= RECOGNITION_MARGIN:
            return self.employee_ids[idx], self.employee_names[idx], best
        return None, "Unknown", best

    def _handle_attendance_event(self, employee_id: Optional[str], name: str) -> Optional[str]:
        if employee_id is None:
            return None

        now = time.time()
        last_try = self.last_attempt_times.get(employee_id, 0.0)
        if now - last_try < ATTENDANCE_COOLDOWN_SECONDS:
            return None
        self.last_attempt_times[employee_id] = now

        if employee_id in self.marked_today:
            return f"Already marked today: {name}"

        inserted = self.attendance.mark_now(employee_id)
        if inserted:
            self.marked_today.add(employee_id)
            self.logger.info("Attendance marked for %s (%s)", name, employee_id)
            return f"Attendance marked: {name}"

        self.marked_today.add(employee_id)
        return f"Already marked today: {name}"

    def _rollover_day_if_needed(self) -> None:
        today = date.today().isoformat()
        if today != self.current_day:
            self.current_day = today
            self.marked_today = self.db.today_marked_employee_ids()
            self.last_attempt_times.clear()
            self.logger.info("Date changed. Attendance cache refreshed for %s.", today)

    @staticmethod
    def _draw_face_box(frame: np.ndarray, box: np.ndarray, label: str, score: float, known: bool) -> None:
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (30, 180, 30) if known else (20, 20, 220)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_status_bar(frame: np.ndarray, status_events: List[str]) -> None:
        if not status_events:
            return

        latest = status_events[-1]
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (35, 35, 35), -1)
        cv2.putText(
            frame,
            latest,
            (20, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
