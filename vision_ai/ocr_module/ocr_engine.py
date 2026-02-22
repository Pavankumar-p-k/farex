from __future__ import annotations

from typing import Sequence
import time

import cv2
import numpy as np
import torch

from vision_ai.utils.types import OCRItem, OCRResult

try:
    import easyocr
except ImportError:  # pragma: no cover - handled at runtime.
    easyocr = None


class RealTimeOCREngine:
    def __init__(
        self,
        languages: Sequence[str],
        confidence: float = 0.35,
        prefer_gpu: bool = True,
        multi_pass: bool = True,
        upscale_factor: float = 1.5,
        temporal_seconds: float = 1.8,
        max_entries: int = 90,
    ) -> None:
        if easyocr is None:
            raise RuntimeError("easyocr is required. Install with: pip install easyocr")
        self.confidence = confidence
        self.use_gpu = bool(prefer_gpu and torch.cuda.is_available())
        self.multi_pass = bool(multi_pass)
        self.upscale_factor = max(1.0, float(upscale_factor))
        self.temporal_seconds = max(0.0, float(temporal_seconds))
        self.max_entries = max(20, int(max_entries))
        self.reader = easyocr.Reader(lang_list=list(languages), gpu=self.use_gpu, verbose=False)
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self._recent_entries: list[tuple[float, OCRItem]] = []

    @property
    def device_name(self) -> str:
        if not self.use_gpu:
            return "cpu"
        try:
            return f"cuda:{torch.cuda.get_device_name(0)}"
        except Exception:
            return "cuda"

    def detect(self, frame_id: int, frame_bgr: np.ndarray) -> OCRResult:
        enhanced = self._enhance(frame_bgr)
        variants: list[tuple[np.ndarray, float]] = [(enhanced, 1.0)]
        if self.multi_pass:
            variants.append((self._threshold_variant(enhanced), 1.0))
            if self.upscale_factor > 1.01:
                variants.append((self._upscale_variant(enhanced, self.upscale_factor), self.upscale_factor))

        candidates: list[OCRItem] = []
        for image, scale in variants:
            candidates.extend(self._read_variant(image=image, scale=scale))

        merged = self._merge_entries(candidates)
        merged = self._apply_temporal_stability(merged)
        if len(merged) > self.max_entries:
            merged = sorted(merged, key=lambda item: item.confidence, reverse=True)[: self.max_entries]
        return OCRResult(frame_id=frame_id, entries=merged)

    def _read_variant(self, image: np.ndarray, scale: float) -> list[OCRItem]:
        try:
            raw = self.reader.readtext(
                image,
                detail=1,
                paragraph=False,
                batch_size=8 if self.use_gpu else 1,
                rotation_info=[90, 180, 270],
                text_threshold=0.55,
                low_text=0.25,
                link_threshold=0.30,
            )
        except Exception:
            return []

        entries: list[OCRItem] = []
        inv_scale = 1.0 / max(1e-6, scale)
        min_conf = max(0.10, self.confidence * 0.75)
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            box_raw, text_raw, conf_raw = item
            text = " ".join(str(text_raw).strip().split())
            if not text:
                continue
            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                continue
            if conf < min_conf:
                continue

            polygon: list[tuple[int, int]] = []
            for p in box_raw:
                vals = list(p)
                if len(vals) < 2:
                    continue
                x = int(round(float(vals[0]) * inv_scale))
                y = int(round(float(vals[1]) * inv_scale))
                polygon.append((x, y))
            if len(polygon) < 4:
                continue
            entries.append(OCRItem(text=text, confidence=conf, polygon=polygon))
        return entries

    def _enhance(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_luma = float(np.mean(gray))
        boosted = frame
        if mean_luma < 95.0:
            gamma = max(0.45, min(0.9, mean_luma / 95.0))
            lut = np.array([((v / 255.0) ** gamma) * 255.0 for v in range(256)], dtype=np.uint8)
            boosted = cv2.LUT(frame, lut)

        lab = cv2.cvtColor(boosted, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        merged = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
        return cv2.addWeighted(enhanced, 1.2, blurred, -0.2, 0)

    @staticmethod
    def _threshold_variant(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            25,
            11,
        )
        cleaned = cv2.medianBlur(adaptive, 3)
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _upscale_variant(frame: np.ndarray, factor: float) -> np.ndarray:
        h, w = frame.shape[:2]
        return cv2.resize(
            frame,
            (max(2, int(round(w * factor))), max(2, int(round(h * factor)))),
            interpolation=cv2.INTER_CUBIC,
        )

    def _merge_entries(self, entries: list[OCRItem]) -> list[OCRItem]:
        if not entries:
            return []
        ordered = sorted(entries, key=lambda item: item.confidence, reverse=True)
        merged: list[OCRItem] = []
        for candidate in ordered:
            candidate_key = candidate.text.casefold()
            combined = False
            for idx, existing in enumerate(merged):
                existing_key = existing.text.casefold()
                iou = self._polygon_iou(existing.polygon, candidate.polygon)
                if existing_key == candidate_key and iou >= 0.20:
                    merged[idx] = OCRItem(
                        text=existing.text if len(existing.text) >= len(candidate.text) else candidate.text,
                        confidence=float(min(1.0, max(existing.confidence, candidate.confidence) + 0.03)),
                        polygon=self._smooth_polygon(existing.polygon, candidate.polygon, alpha=0.5),
                    )
                    combined = True
                    break
                if iou >= 0.72 and candidate.confidence <= existing.confidence:
                    combined = True
                    break
            if not combined:
                merged.append(candidate)
        return [item for item in merged if item.confidence >= self.confidence]

    def _apply_temporal_stability(self, entries: list[OCRItem]) -> list[OCRItem]:
        if not entries or self.temporal_seconds <= 0.0:
            return entries

        now = time.perf_counter()
        self._recent_entries = [
            (ts, item)
            for ts, item in self._recent_entries
            if (now - ts) <= self.temporal_seconds
        ]

        stable: list[OCRItem] = []
        for item in entries:
            best_prev: OCRItem | None = None
            best_iou = 0.0
            key = item.text.casefold()
            for _, previous in self._recent_entries:
                if previous.text.casefold() != key:
                    continue
                iou = self._polygon_iou(item.polygon, previous.polygon)
                if iou > best_iou:
                    best_iou = iou
                    best_prev = previous

            if best_prev is not None and best_iou >= 0.18:
                stable.append(
                    OCRItem(
                        text=item.text,
                        confidence=float(min(1.0, max(item.confidence, best_prev.confidence * 0.99))),
                        polygon=self._smooth_polygon(item.polygon, best_prev.polygon, alpha=0.78),
                    )
                )
            else:
                stable.append(item)

        for item in stable:
            self._recent_entries.append((now, item))
        if len(self._recent_entries) > 450:
            self._recent_entries = self._recent_entries[-450:]
        return stable

    @staticmethod
    def _smooth_polygon(
        current: list[tuple[int, int]],
        previous: list[tuple[int, int]],
        alpha: float = 0.75,
    ) -> list[tuple[int, int]]:
        if len(current) < 4 or len(previous) < 4:
            return current
        out: list[tuple[int, int]] = []
        total = min(len(current), len(previous))
        for idx in range(total):
            cx, cy = current[idx]
            px, py = previous[idx]
            x = int(round((alpha * cx) + ((1.0 - alpha) * px)))
            y = int(round((alpha * cy) + ((1.0 - alpha) * py)))
            out.append((x, y))
        return out if len(out) >= 4 else current

    @staticmethod
    def _polygon_iou(poly_a: list[tuple[int, int]], poly_b: list[tuple[int, int]]) -> float:
        ax1, ay1, ax2, ay2 = RealTimeOCREngine._polygon_bounds(poly_a)
        bx1, by1, bx2, by2 = RealTimeOCREngine._polygon_bounds(poly_b)
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = float((ix2 - ix1) * (iy2 - iy1))
        area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
        area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
        return inter / (area_a + area_b - inter + 1e-6)

    @staticmethod
    def _polygon_bounds(polygon: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        if not polygon:
            return (0, 0, 1, 1)
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return min(xs), min(ys), max(xs), max(ys)
