from typing import List, Optional, Tuple

import cv2
import numpy as np

from .camera import CameraStream
from .config import RECOGNITION_MARGIN, RECOGNITION_THRESHOLD, SINGLE_USER_RECOGNITION_THRESHOLD
from .database import AttendanceDatabase, EmployeeRecord
from .exceptions import AttendanceError
from .face_engine import FaceEngine
from .logger import setup_logger


class FaceRecognitionService:
    def __init__(self, db: AttendanceDatabase, engine: FaceEngine, threshold: float = RECOGNITION_THRESHOLD):
        self.db = db
        self.engine = engine
        self.threshold = threshold
        self.logger = setup_logger(self.__class__.__name__)

        self.employee_ids: List[str] = []
        self.employee_names: List[str] = []
        self.known_matrix: Optional[np.ndarray] = None
        self._refresh_known_faces()

    def run(self, camera_index: int = 0) -> None:
        if self.known_matrix is None or not self.employee_ids:
            raise AttendanceError("No registered people found. Run registration first.")

        window_name = "Live Face Recognition - Press Q to exit"
        self.logger.info("Starting live face recognition")

        with CameraStream(camera_index) as cam:
            while True:
                frame = cam.read()
                batch = self.engine.extract_embeddings(frame)
                labels: List[str] = []

                for emb, box in zip(batch.embeddings, batch.boxes):
                    _, name, score = self._match_face(emb)
                    labels.append(name)
                    self._draw_face_box(frame, box, name, score, name != "Unknown")

                self._draw_status(frame, labels)
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
    def _draw_status(frame: np.ndarray, labels: List[str]) -> None:
        if not labels:
            message = "No face recognized"
        else:
            message = "Detected: " + ", ".join(labels)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (35, 35, 35), -1)
        cv2.putText(
            frame,
            message,
            (20, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
