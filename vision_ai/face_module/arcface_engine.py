from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import cv2

try:
    from insightface.app import FaceAnalysis
except ImportError:  # pragma: no cover - handled at runtime.
    FaceAnalysis = None


@dataclass
class ArcFaceDetection:
    bbox: tuple[int, int, int, int]
    score: float
    embedding: np.ndarray


class ArcFaceEngine:
    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        prefer_gpu: bool = True,
    ) -> None:
        self.use_gpu = bool(prefer_gpu and torch.cuda.is_available())
        self._backend = "insightface"
        self._haar = None
        self.app = None

        if FaceAnalysis is not None:
            providers: Iterable[str]
            if self.use_gpu:
                providers = ("CUDAExecutionProvider", "CPUExecutionProvider")
            else:
                providers = ("CPUExecutionProvider",)
            try:
                self.app = FaceAnalysis(name=model_name, providers=list(providers))
                self.app.prepare(ctx_id=0 if self.use_gpu else -1, det_size=det_size)
            except Exception:
                self.app = None

        if self.app is None:
            # Compatibility fallback for environments where insightface wheel/build is unavailable.
            self._backend = "opencv-haarcascade"
            self.use_gpu = False
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)
            if self._haar.empty():
                raise RuntimeError("Failed to initialize OpenCV Haar face detector.")

    @property
    def device_name(self) -> str:
        if self._backend != "insightface":
            return "cpu-fallback"
        if not self.use_gpu:
            return "cpu"
        try:
            return f"cuda:{torch.cuda.get_device_name(0)}"
        except Exception:
            return "cuda"

    def detect(self, frame_bgr: np.ndarray) -> list[ArcFaceDetection]:
        if self._backend != "insightface":
            return self._detect_fallback(frame_bgr)

        faces = self.app.get(frame_bgr)
        output: list[ArcFaceDetection] = []
        for face in faces:
            bbox_raw = [int(v) for v in face.bbox.tolist()]
            x1, y1, x2, y2 = bbox_raw[0], bbox_raw[1], bbox_raw[2], bbox_raw[3]

            embedding = np.asarray(face.embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm

            output.append(
                ArcFaceDetection(
                    bbox=(x1, y1, x2, y2),
                    score=float(getattr(face, "det_score", 1.0)),
                    embedding=embedding,
                )
            )

        output.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]), reverse=True)
        return output

    def _detect_fallback(self, frame_bgr: np.ndarray) -> list[ArcFaceDetection]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        detections = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(60, 60),
        )

        output: list[ArcFaceDetection] = []
        for (x, y, w, h) in detections:
            x1, y1 = int(max(0, x)), int(max(0, y))
            x2 = int(min(frame_bgr.shape[1], x + w))
            y2 = int(min(frame_bgr.shape[0], y + h))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame_bgr[y1:y2, x1:x2]
            embedding = self._embed_crop(crop)
            output.append(
                ArcFaceDetection(
                    bbox=(x1, y1, x2, y2),
                    score=1.0,
                    embedding=embedding,
                )
            )

        output.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]), reverse=True)
        return output

    @staticmethod
    def _embed_crop(crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr.size == 0:
            return np.zeros(512, dtype=np.float32)

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        # Keep 512-dim to remain DB-compatible with ArcFace-style vectors.
        resized = cv2.resize(gray, (32, 16), interpolation=cv2.INTER_AREA)
        embedding = resized.reshape(-1).astype(np.float32) / 255.0
        embedding -= float(np.mean(embedding))
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding /= norm
        return embedding
