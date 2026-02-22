from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as f
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from .config import DEVICE, FACE_DETECTION_THRESHOLD, MIN_FACE_SIZE
from .exceptions import FaceEngineError

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - runtime dependency guard
    mp = None


@dataclass
class FaceBatch:
    embeddings: List[np.ndarray]
    boxes: List[np.ndarray]
    confidences: List[float]


class FaceEngine:
    def __init__(
        self,
        device: str = DEVICE,
        detection_threshold: float = FACE_DETECTION_THRESHOLD,
        min_face_size: int = MIN_FACE_SIZE,
    ):
        if mp is None:
            raise FaceEngineError("mediapipe is required. Install dependencies from requirements.txt.")

        self.device = torch.device(device)
        self.detection_threshold = detection_threshold
        self.min_face_size = min_face_size

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        try:
            self.mp_face = mp.solutions.face_detection
            self.detector = self.mp_face.FaceDetection(
                model_selection=0,
                min_detection_confidence=detection_threshold,
            )

            weights = ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
            backbone.fc = torch.nn.Identity()
            self.embedder = backbone.eval().to(self.device)

            self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
            self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        except Exception as exc:
            raise FaceEngineError(f"Failed to initialize face models: {exc}") from exc

    def extract_embeddings(self, frame: np.ndarray) -> FaceBatch:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.detector.process(rgb)
        except Exception as exc:
            raise FaceEngineError(f"Face extraction failed: {exc}") from exc

        if not result.detections:
            return FaceBatch(embeddings=[], boxes=[], confidences=[])

        h, w = frame.shape[:2]
        crops = []
        boxes = []
        confs = []

        for det in result.detections:
            score = float(det.score[0]) if det.score else 0.0
            if score < self.detection_threshold:
                continue

            rel = det.location_data.relative_bounding_box
            x1 = max(0, int(rel.xmin * w))
            y1 = max(0, int(rel.ymin * h))
            bw = int(rel.width * w)
            bh = int(rel.height * h)
            x2 = min(w, x1 + bw)
            y2 = min(h, y1 + bh)

            if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
                continue

            crop, stable_box = self._extract_stable_crop(rgb, x1, y1, x2, y2)
            if crop.size == 0:
                continue

            crops.append(crop)
            boxes.append(stable_box)
            confs.append(score)

        if not crops:
            return FaceBatch(embeddings=[], boxes=[], confidences=[])

        try:
            tensor_batch = self._to_tensor_batch(crops)
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        raw = self.embedder(tensor_batch)
                else:
                    raw = self.embedder(tensor_batch)

                normed = f.normalize(raw, p=2, dim=1)
                emb = normed.detach().cpu().numpy().astype(np.float32)
        except Exception as exc:
            raise FaceEngineError(f"Embedding generation failed: {exc}") from exc

        return FaceBatch(
            embeddings=[emb[i] for i in range(emb.shape[0])],
            boxes=boxes,
            confidences=confs,
        )

    def _to_tensor_batch(self, face_crops: List[np.ndarray]) -> torch.Tensor:
        processed = []
        for crop in face_crops:
            tensor = torch.from_numpy(self._preprocess_crop(crop)).permute(2, 0, 1).float() / 255.0
            processed.append(tensor)

        batch = torch.stack(processed, dim=0).to(self.device)
        batch = (batch - self.mean) / self.std
        return batch

    def _extract_stable_crop(
        self,
        rgb: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = rgb.shape[:2]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        side = int(max(bw, bh) * 1.05)
        cx = int((x1 + x2) * 0.5)
        cy = int((y1 + y2) * 0.5)

        sx1 = max(0, cx - side // 2)
        sy1 = max(0, cy - side // 2)
        sx2 = min(w, sx1 + side)
        sy2 = min(h, sy1 + side)

        if sx2 <= sx1 or sy2 <= sy1:
            return np.empty((0, 0, 3), dtype=rgb.dtype), np.array([x1, y1, x2, y2], dtype=np.float32)

        stable_box = np.array([sx1, sy1, sx2, sy2], dtype=np.float32)
        return rgb[sy1:sy2, sx1:sx2], stable_box

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        if crop.shape[0] < 224 or crop.shape[1] < 224:
            resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
        else:
            resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)

        # Normalize illumination and suppress background corners.
        ycrcb = cv2.cvtColor(resized, cv2.COLOR_RGB2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
        y_channel = self.clahe.apply(y_channel)
        balanced = cv2.cvtColor(
            cv2.merge([y_channel, cr_channel, cb_channel]),
            cv2.COLOR_YCrCb2RGB,
        )

        mask = np.zeros((224, 224), dtype=np.float32)
        cv2.ellipse(mask, (112, 112), (84, 100), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=6.0, sigmaY=6.0)
        mask = mask[..., None]

        balanced_f = balanced.astype(np.float32)
        mean_color = balanced_f.mean(axis=(0, 1), keepdims=True)
        focused = (balanced_f * mask) + (mean_color * (1.0 - mask))
        return np.clip(focused, 0.0, 255.0).astype(np.uint8)
