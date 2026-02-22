from __future__ import annotations

from pathlib import Path
from typing import Sequence
import time

import numpy as np
import torch

from vision_ai.utils.types import ObjectItem, ObjectResult

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - handled at runtime.
    YOLO = None


class YOLODetector:
    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        classes: Sequence[int] | None = None,
        prefer_gpu: bool = True,
        max_detections: int = 300,
        ensemble_enabled: bool = True,
        ensemble_img_size: int = 960,
        tta_enabled: bool = True,
        temporal_seconds: float = 0.65,
    ) -> None:
        if YOLO is None:
            raise RuntimeError("ultralytics is required. Install with: pip install ultralytics")

        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.max_detections = max(20, int(max_detections))
        self.ensemble_enabled = bool(ensemble_enabled)
        self.ensemble_img_size = max(512, int(ensemble_img_size))
        self.tta_enabled = bool(tta_enabled)
        self.temporal_seconds = max(0.0, float(temporal_seconds))
        self.classes = list(classes) if classes else None
        self.use_gpu = bool(prefer_gpu and torch.cuda.is_available())
        self.device = "cuda:0" if self.use_gpu else "cpu"
        self.half = self.use_gpu
        self.model = YOLO(model_path)
        self.model_path = model_path
        self._history: list[tuple[float, list[ObjectItem]]] = []

    @property
    def device_name(self) -> str:
        if not self.use_gpu:
            return "cpu"
        try:
            return f"cuda:{torch.cuda.get_device_name(0)}"
        except Exception:
            return "cuda"

    @property
    def model_name(self) -> str:
        return Path(self.model_path).name

    def detect(self, frame_id: int, frame_bgr: np.ndarray) -> ObjectResult:
        detections = self._predict_items(frame_bgr, image_size=self.image_size, use_tta=self.tta_enabled)
        if self.ensemble_enabled and self.ensemble_img_size != self.image_size:
            second_pass = self._predict_items(
                frame_bgr,
                image_size=self.ensemble_img_size,
                use_tta=False,
            )
            detections = self._fuse_detections(detections + second_pass, iou_threshold=0.55)

        detections = self._temporal_refine(detections)
        if len(detections) > self.max_detections:
            detections = sorted(detections, key=lambda item: item.confidence, reverse=True)[: self.max_detections]
        return ObjectResult(frame_id=frame_id, detections=detections)

    def _predict_items(
        self,
        frame_bgr: np.ndarray,
        image_size: int,
        use_tta: bool,
    ) -> list[ObjectItem]:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.confidence,
            iou=self.iou,
            imgsz=image_size,
            classes=self.classes,
            device=self.device,
            half=self.half,
            verbose=False,
            max_det=self.max_detections,
            augment=use_tta,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return []

        names = result.names
        detections: list[ObjectItem] = []
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            if isinstance(names, dict):
                label = str(names.get(class_id, class_id))
            elif isinstance(names, (list, tuple)) and class_id < len(names):
                label = str(names[class_id])
            else:
                label = str(class_id)

            detections.append(
                ObjectItem(
                    label=label,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections

    def _temporal_refine(self, current: list[ObjectItem]) -> list[ObjectItem]:
        if self.temporal_seconds <= 0.0:
            return current

        now = time.perf_counter()
        self._history = [
            (ts, items)
            for ts, items in self._history
            if (now - ts) <= self.temporal_seconds
        ]

        previous_items: list[ObjectItem] = []
        if self._history:
            previous_items = self._history[-1][1]

        refined: list[ObjectItem] = []
        for item in current:
            best_prev: ObjectItem | None = None
            best_iou = 0.0
            for prev in previous_items:
                if prev.label != item.label:
                    continue
                iou = self._bbox_iou(item.bbox, prev.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_prev = prev

            if best_prev is not None and best_iou >= 0.35:
                x1 = int(round((0.75 * item.bbox[0]) + (0.25 * best_prev.bbox[0])))
                y1 = int(round((0.75 * item.bbox[1]) + (0.25 * best_prev.bbox[1])))
                x2 = int(round((0.75 * item.bbox[2]) + (0.25 * best_prev.bbox[2])))
                y2 = int(round((0.75 * item.bbox[3]) + (0.25 * best_prev.bbox[3])))
                confidence = max(item.confidence, best_prev.confidence * 0.98)
                refined.append(
                    ObjectItem(
                        label=item.label,
                        confidence=float(min(1.0, confidence)),
                        bbox=(x1, y1, x2, y2),
                    )
                )
            else:
                refined.append(item)

        refined = self._fuse_detections(refined, iou_threshold=0.6)
        self._history.append((now, refined))
        if len(self._history) > 6:
            self._history = self._history[-6:]
        return refined

    @staticmethod
    def _fuse_detections(items: list[ObjectItem], iou_threshold: float = 0.55) -> list[ObjectItem]:
        if not items:
            return []
        ordered = sorted(items, key=lambda item: item.confidence, reverse=True)
        fused: list[ObjectItem] = []
        for candidate in ordered:
            merged = False
            for idx, existing in enumerate(fused):
                if existing.label != candidate.label:
                    continue
                if YOLODetector._bbox_iou(existing.bbox, candidate.bbox) < iou_threshold:
                    continue
                weight_a = max(1e-4, existing.confidence)
                weight_b = max(1e-4, candidate.confidence)
                denom = weight_a + weight_b
                merged_bbox = (
                    int(round((existing.bbox[0] * weight_a + candidate.bbox[0] * weight_b) / denom)),
                    int(round((existing.bbox[1] * weight_a + candidate.bbox[1] * weight_b) / denom)),
                    int(round((existing.bbox[2] * weight_a + candidate.bbox[2] * weight_b) / denom)),
                    int(round((existing.bbox[3] * weight_a + candidate.bbox[3] * weight_b) / denom)),
                )
                fused[idx] = ObjectItem(
                    label=existing.label,
                    confidence=float(max(existing.confidence, candidate.confidence)),
                    bbox=merged_bbox,
                )
                merged = True
                break
            if not merged:
                fused.append(candidate)
        return fused

    @staticmethod
    def _bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
        area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
        area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
        return inter_area / (area_a + area_b - inter_area + 1e-6)
