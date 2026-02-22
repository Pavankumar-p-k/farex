from datetime import datetime
from typing import List

import cv2
import numpy as np

from .camera import CameraStream
from .config import DUPLICATE_FACE_SIMILARITY_THRESHOLD, REGISTRATION_SAMPLES, SAMPLE_EVERY_N_FRAMES
from .database import AttendanceDatabase, EmployeeRecord
from .exceptions import AttendanceError
from .face_engine import FaceEngine
from .logger import setup_logger


class RegistrationService:
    def __init__(self, db: AttendanceDatabase, engine: FaceEngine):
        self.db = db
        self.engine = engine
        self.logger = setup_logger(self.__class__.__name__)

    def register_employee(
        self,
        employee_id: str,
        name: str,
        camera_index: int = 0,
        target_samples: int = REGISTRATION_SAMPLES,
        sample_every_n_frames: int = SAMPLE_EVERY_N_FRAMES,
    ) -> None:
        if not employee_id.strip():
            raise AttendanceError("employee_id cannot be empty.")
        if not name.strip():
            raise AttendanceError("name cannot be empty.")
        if target_samples < 5:
            raise AttendanceError("target_samples should be at least 5.")

        collected: List[np.ndarray] = []
        frame_index = 0
        start_time = datetime.now()
        window_name = "Registration - Press Q to cancel"

        self.logger.info("Starting registration for %s (%s)", employee_id, name)

        with CameraStream(camera_index) as cam:
            while len(collected) < target_samples:
                frame = cam.read()
                frame_index += 1

                try:
                    batch = self.engine.extract_embeddings(frame)
                except AttendanceError:
                    raise
                except Exception as exc:
                    self.logger.exception("Face extraction error during registration.")
                    raise AttendanceError(f"Registration failed due to face extraction error: {exc}") from exc

                status = "Align your face with the camera"
                if len(batch.embeddings) == 1:
                    if frame_index % max(1, sample_every_n_frames) == 0:
                        collected.append(batch.embeddings[0])
                        status = f"Captured sample {len(collected)}/{target_samples}"
                    else:
                        status = "Hold still..."
                elif len(batch.embeddings) > 1:
                    status = "Only one face should be visible"
                else:
                    status = "No face detected"

                self._draw_registration_overlay(frame, batch.boxes, status, len(collected), target_samples)
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    raise AttendanceError("Registration cancelled by user.")

        if len(collected) < target_samples:
            raise AttendanceError("Insufficient samples captured for registration.")

        avg_encoding = self._average_encoding(collected)
        self._validate_identity_uniqueness(avg_encoding, employee_id)
        self.db.upsert_employee(employee_id=employee_id, name=name, encoding=avg_encoding)

        elapsed = (datetime.now() - start_time).total_seconds()
        self.logger.info(
            "Employee %s registered successfully with %d samples in %.1fs",
            employee_id,
            len(collected),
            elapsed,
        )

    @staticmethod
    def _average_encoding(embeddings: List[np.ndarray]) -> np.ndarray:
        matrix = np.vstack(embeddings).astype(np.float32)
        vector = matrix.mean(axis=0)
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            raise AttendanceError("Unable to normalize average encoding.")
        return vector / norm

    def _validate_identity_uniqueness(self, encoding: np.ndarray, employee_id: str) -> None:
        records: List[EmployeeRecord] = self.db.list_employees()
        if not records:
            return

        query = encoding.astype(np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-9)

        for record in records:
            if record.employee_id == employee_id:
                continue
            known = record.encoding.astype(np.float32)
            known = known / max(float(np.linalg.norm(known)), 1e-9)
            similarity = float(known @ query)
            if similarity >= DUPLICATE_FACE_SIMILARITY_THRESHOLD:
                raise AttendanceError(
                    f"Captured face is too similar to existing user '{record.name}' ({record.employee_id}). "
                    "Use a different person or capture cleaner samples."
                )

    @staticmethod
    def _draw_registration_overlay(
        frame: np.ndarray,
        boxes: List[np.ndarray],
        status: str,
        collected: int,
        target_samples: int,
    ) -> None:
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Samples: {collected}/{target_samples}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
