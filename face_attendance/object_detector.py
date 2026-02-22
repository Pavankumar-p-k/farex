from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .exceptions import DetectorError
from .object_types import ObjectDetection

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - handled at runtime.
    YOLO = None


class YoloV8Detector:
    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        prefer_gpu: bool = True,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: Optional[Sequence[int]] = None,
        img_size: int = 640,
        max_detections: int = 300,
    ) -> None:
        if YOLO is None:
            raise DetectorError(
                "Ultralytics YOLO is not installed. Install dependencies with: pip install -r requirements.txt"
            )

        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = list(classes) if classes else None
        self.img_size = img_size
        self.max_detections = max_detections

        self._gpu_enabled = bool(prefer_gpu and torch.cuda.is_available())
        self._device = "cuda:0" if self._gpu_enabled else "cpu"
        self._half = self._gpu_enabled

        try:
            self.model = YOLO(self.model_path)
        except Exception as exc:
            raise DetectorError(f"Failed to initialize YOLO model '{self.model_path}': {exc}") from exc

    @property
    def device_name(self) -> str:
        if not self._gpu_enabled:
            return "cpu"
        try:
            return f"cuda:{torch.cuda.get_device_name(0)}"
        except Exception:
            return "cuda"

    @property
    def model_name(self) -> str:
        return Path(self.model_path).name

    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self.classes,
                imgsz=self.img_size,
                max_det=self.max_detections,
                device=self._device,
                half=self._half,
                verbose=False,
            )
        except Exception as exc:
            raise DetectorError(f"YOLO inference failed: {exc}") from exc

        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return []

        names = result.names
        detections: list[ObjectDetection] = []
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = (int(v) for v in xyxy)

            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            if isinstance(names, dict):
                label = str(names.get(class_id, class_id))
            elif isinstance(names, (list, tuple)) and class_id < len(names):
                label = str(names[class_id])
            else:
                label = str(class_id)

            detections.append(
                ObjectDetection(
                    class_id=class_id,
                    label=label,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                )
            )

        return detections
