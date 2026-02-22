from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from .exceptions import OCRError
from .ocr_types import OCRDetection, Polygon

try:
    import easyocr
except ImportError:  # pragma: no cover - handled at runtime.
    easyocr = None


class EasyOCREngine:
    def __init__(
        self,
        languages: Sequence[str],
        prefer_gpu: bool = True,
        min_confidence: float = 0.35,
        retry_confidence: float = 0.55,
    ) -> None:
        if easyocr is None:
            raise OCRError(
                "EasyOCR is not installed. Install dependencies with: pip install -r requirements.txt"
            )

        self.languages = list(languages)
        self.min_confidence = min_confidence
        self.retry_confidence = retry_confidence
        self.gpu_enabled = bool(prefer_gpu and torch.cuda.is_available())

        if self.gpu_enabled:
            torch.backends.cudnn.benchmark = True

        try:
            self.reader = easyocr.Reader(
                lang_list=self.languages,
                gpu=self.gpu_enabled,
                verbose=False,
            )
        except Exception as exc:
            raise OCRError(f"Failed to initialize EasyOCR reader: {exc}") from exc

    @property
    def device_name(self) -> str:
        if not self.gpu_enabled:
            return "cpu"
        try:
            return f"cuda:{torch.cuda.get_device_name(0)}"
        except Exception:
            return "cuda"

    def detect_and_recognize(self, frame: np.ndarray) -> list[OCRDetection]:
        try:
            raw_results = self.reader.readtext(
                frame,
                detail=1,
                paragraph=False,
                batch_size=8 if self.gpu_enabled else 1,
                canvas_size=2048,
                mag_ratio=1.2,
                text_threshold=0.6,
                low_text=0.3,
                link_threshold=0.35,
                slope_ths=0.35,
                ycenter_ths=0.7,
                rotation_info=[90, 180, 270],
                decoder="greedy",
            )
        except Exception as exc:
            raise OCRError(f"OCR inference failed: {exc}") from exc

        detections: list[OCRDetection] = []
        for item in raw_results:
            parsed = self._parse_readtext_item(item)
            if parsed is None:
                continue

            polygon, text, confidence = parsed
            if confidence < self.retry_confidence:
                retried = self._retry_with_rectification(frame, polygon)
                if retried is not None:
                    retried_text, retried_score = retried
                    if retried_score > confidence and retried_text:
                        text = retried_text
                        confidence = retried_score

            if confidence < self.min_confidence:
                continue

            detections.append(
                OCRDetection(
                    polygon=polygon,
                    text=text,
                    confidence=confidence,
                    angle_degrees=self._estimate_angle(polygon),
                )
            )

        return detections

    def _parse_readtext_item(
        self, item: Sequence[object]
    ) -> Optional[Tuple[Polygon, str, float]]:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            return None

        polygon = self._to_polygon(item[0])
        if not polygon:
            return None

        text = str(item[1]).strip()
        if not text:
            return None

        try:
            confidence = float(item[2])
        except (TypeError, ValueError):
            return None

        return polygon, text, confidence

    def _retry_with_rectification(
        self, frame: np.ndarray, polygon: Polygon
    ) -> Optional[Tuple[str, float]]:
        cropped = self._perspective_crop(frame, polygon)
        if cropped is None:
            return None

        try:
            retries = self.reader.readtext(
                cropped,
                detail=1,
                paragraph=False,
                batch_size=1,
                contrast_ths=0.05,
                adjust_contrast=0.7,
                rotation_info=[90, 180, 270],
                decoder="greedy",
            )
        except Exception:
            return None

        best_text = ""
        best_score = 0.0
        for entry in retries:
            parsed = self._parse_readtext_item(entry)
            if parsed is None:
                continue
            _, text, score = parsed
            if score > best_score:
                best_text = text
                best_score = score

        if not best_text:
            return None
        return best_text, best_score

    @staticmethod
    def _to_polygon(raw_bbox: object) -> Polygon:
        if not isinstance(raw_bbox, Iterable):
            return []

        points: Polygon = []
        for point in raw_bbox:
            if not isinstance(point, Iterable):
                continue
            values = list(point)
            if len(values) < 2:
                continue
            x, y = int(values[0]), int(values[1])
            points.append((x, y))

        return points if len(points) >= 4 else []

    @staticmethod
    def _estimate_angle(polygon: Polygon) -> float:
        if len(polygon) < 2:
            return 0.0

        x1, y1 = polygon[0]
        x2, y2 = polygon[1]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if angle > 90.0:
            angle -= 180.0
        if angle < -90.0:
            angle += 180.0
        return angle

    def _perspective_crop(self, frame: np.ndarray, polygon: Polygon) -> Optional[np.ndarray]:
        if len(polygon) < 4:
            return None

        points = np.array(polygon[:4], dtype=np.float32)
        rect = self._order_points(points)

        width_top = np.linalg.norm(rect[1] - rect[0])
        width_bottom = np.linalg.norm(rect[2] - rect[3])
        max_width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(rect[3] - rect[0])
        height_right = np.linalg.norm(rect[2] - rect[1])
        max_height = int(max(height_left, height_right))

        if max_width < 12 or max_height < 12:
            return None

        destination = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(rect, destination)
        return cv2.warpPerspective(frame, matrix, (max_width, max_height))

    @staticmethod
    def _order_points(points: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).reshape(-1)

        rect[0] = points[np.argmin(sums)]  # top-left
        rect[2] = points[np.argmax(sums)]  # bottom-right
        rect[1] = points[np.argmin(diffs)]  # top-right
        rect[3] = points[np.argmax(diffs)]  # bottom-left
        return rect
