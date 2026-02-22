from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .camera_capture import open_camera_capture
from .logger import setup_logger
from .ocr_engine import EasyOCREngine
from .ocr_persistence import OCRTextSaver
from .ocr_preprocessor import FramePreprocessor
from .ocr_types import OCRDetection


@dataclass
class OCRRuntimeConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    frame_fps: int = 30
    detection_interval: int = 2
    languages: Sequence[str] = ("en",)
    prefer_gpu: bool = True
    min_confidence: float = 0.35
    output_file: Path = Path("logs/ocr_detected_text.txt")
    benchmark_seconds: int = 0


class RealTimeOCRSystem:
    def __init__(self, config: OCRRuntimeConfig) -> None:
        self.config = config
        self.logger = setup_logger("ocr")
        self.preprocessor = FramePreprocessor()
        self.engine = EasyOCREngine(
            languages=config.languages,
            prefer_gpu=config.prefer_gpu,
            min_confidence=config.min_confidence,
        )
        self.saver = OCRTextSaver(
            output_path=config.output_file,
            min_confidence=config.min_confidence,
        )
        self._last_detections: list[OCRDetection] = []
        self._fps = 0.0
        self._fps_counter = 0
        self._fps_start = time.perf_counter()
        self._status_message = ""
        self._status_until = 0.0

    def run(self) -> dict[str, float] | None:
        cap, backend_name = open_camera_capture(self.config.camera_index)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.frame_fps)
        cv2.setUseOptimized(True)

        window_name = "Real-Time OCR (press Q to quit, S to save text)"
        self.logger.info("OCR device: %s", self.engine.device_name)
        self.logger.info("Camera backend: %s", backend_name)
        self.logger.info("Saving OCR snapshots to: %s", self.saver.output_path)
        self.logger.info("Controls: press 'S' to save current text, 'Q' to quit.")

        frame_index = 0
        total_frames = 0
        total_detection_ms = 0.0
        total_detection_calls = 0
        started_at = time.perf_counter()
        try:
            while True:
                success, frame = cap.read()
                if not success or frame is None:
                    continue

                processed = self.preprocessor.enhance(frame)
                if frame_index % max(1, self.config.detection_interval) == 0:
                    detection_start = time.perf_counter()
                    self._last_detections = self.engine.detect_and_recognize(processed)
                    total_detection_ms += (time.perf_counter() - detection_start) * 1000.0
                    total_detection_calls += 1

                self._update_fps()
                annotated = self._draw_overlay(frame.copy(), self._last_detections)
                cv2.imshow(window_name, annotated)
                total_frames += 1

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    saved_count = self.saver.save_snapshot(self._last_detections)
                    self._status_message = (
                        f"Saved {saved_count} line(s) to {self.saver.output_path.name}"
                    )
                    self._status_until = time.time() + 2.0

                frame_index += 1
                if self.config.benchmark_seconds > 0:
                    elapsed = time.perf_counter() - started_at
                    if elapsed >= self.config.benchmark_seconds:
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        if self.config.benchmark_seconds > 0:
            duration = max(1e-6, time.perf_counter() - started_at)
            return {
                "duration_seconds": duration,
                "avg_fps": total_frames / duration,
                "avg_detection_ms": total_detection_ms / max(1, total_detection_calls),
                "frames": float(total_frames),
                "detections": float(total_detection_calls),
            }
        return None

    def _draw_overlay(self, frame: np.ndarray, detections: list[OCRDetection]) -> np.ndarray:
        for detection in detections:
            polygon = np.array(detection.polygon, dtype=np.int32)
            if polygon.ndim != 2 or polygon.shape[0] < 4:
                continue

            color = (70, 220, 70) if detection.confidence >= 0.65 else (60, 180, 255)
            cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)

            x = int(np.min(polygon[:, 0]))
            y = int(np.min(polygon[:, 1])) - 8
            y = max(20, y)
            label = f"{detection.text} ({detection.confidence:.2f})"
            cv2.putText(
                frame,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        lines = self._unique_lines(detections)
        panel_height = min(frame.shape[0] - 20, 24 + 22 * (min(len(lines), 10) + 3))
        cv2.rectangle(frame, (10, 10), (520, panel_height), (20, 20, 20), -1)
        cv2.putText(frame, "Live OCR", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"FPS: {self._fps:.1f} | Device: {self.engine.device_name}",
            (18, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        for idx, line in enumerate(lines[:10], start=0):
            cv2.putText(
                frame,
                f"{idx + 1}. {line}",
                (18, 82 + idx * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (180, 250, 180),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            "S: Save text snapshot | Q: Quit",
            (18, frame.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if self._status_message and time.time() < self._status_until:
            cv2.putText(
                frame,
                self._status_message,
                (18, frame.shape[0] - 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return frame

    @staticmethod
    def _unique_lines(detections: list[OCRDetection]) -> list[str]:
        ordered: dict[str, tuple[str, float]] = {}
        for detection in detections:
            text = detection.text.strip()
            if not text:
                continue
            key = text.lower()
            if key not in ordered or detection.confidence > ordered[key][1]:
                ordered[key] = (text, detection.confidence)

        return [f"{text} ({confidence:.2f})" for text, confidence in ordered.values()]

    def _update_fps(self) -> None:
        self._fps_counter += 1
        now = time.perf_counter()
        elapsed = now - self._fps_start
        if elapsed >= 1.0:
            self._fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_start = now
