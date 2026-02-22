from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from .camera_capture import open_camera_capture
from .logger import setup_logger
from .object_detector import YoloV8Detector
from .object_types import ObjectDetection


@dataclass
class ObjectDetectionConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    frame_fps: int = 30
    model_path: str = "yolov8n.pt"
    prefer_gpu: bool = True
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    classes: Optional[Sequence[int]] = None
    image_size: int = 640
    max_detections: int = 300
    benchmark_seconds: int = 0


class ObjectDetectionService:
    def __init__(self, config: ObjectDetectionConfig) -> None:
        self.detector = YoloV8Detector(
            model_path=config.model_path,
            prefer_gpu=config.prefer_gpu,
            conf_threshold=config.conf_threshold,
            iou_threshold=config.iou_threshold,
            classes=config.classes,
            img_size=config.image_size,
            max_detections=config.max_detections,
        )

    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        return self.detector.detect(frame)

    def annotate(self, frame: np.ndarray, detections: list[ObjectDetection]) -> np.ndarray:
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = self._color_for_class(detection.class_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{detection.label} {detection.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(20, y1 - 8)
            cv2.rectangle(
                frame,
                (x1, label_y - label_size[1] - 8),
                (x1 + label_size[0] + 8, label_y + 4),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 4, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        return frame

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[ObjectDetection], float]:
        start = time.perf_counter()
        detections = self.detect(frame)
        latency_ms = (time.perf_counter() - start) * 1000.0
        annotated = self.annotate(frame.copy(), detections)
        return annotated, detections, latency_ms

    @staticmethod
    def _color_for_class(class_id: int) -> tuple[int, int, int]:
        seed = (class_id + 1) * 37
        return (
            int((seed * 29) % 255),
            int((seed * 59) % 255),
            int((seed * 89) % 255),
        )


class ObjectDetectionRuntime:
    def __init__(self, config: ObjectDetectionConfig) -> None:
        self.config = config
        self.logger = setup_logger("object_detection")
        self.service = ObjectDetectionService(config=config)
        self._fps = 0.0
        self._frame_counter = 0
        self._fps_start = time.perf_counter()

    def run(self) -> dict[str, float] | None:
        cap, backend_name = open_camera_capture(self.config.camera_index)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.frame_fps)
        cv2.setUseOptimized(True)

        self.logger.info("YOLO model: %s", self.service.detector.model_name)
        self.logger.info("YOLO device: %s", self.service.detector.device_name)
        self.logger.info("Camera backend: %s", backend_name)

        started_at = time.perf_counter()
        inference_samples = 0
        total_inference_ms = 0.0
        total_frames = 0
        window_name = "YOLOv8 Object Detection (press Q to quit)"

        try:
            while True:
                success, frame = cap.read()
                if not success or frame is None:
                    continue

                annotated, detections, latency_ms = self.service.process_frame(frame)
                inference_samples += 1
                total_inference_ms += latency_ms
                total_frames += 1
                self._update_fps()

                self._draw_hud(
                    annotated,
                    detections_count=len(detections),
                    latency_ms=latency_ms,
                )
                cv2.imshow(window_name, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

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
                "avg_inference_ms": total_inference_ms / max(1, inference_samples),
                "frames": float(total_frames),
            }
        return None

    def _draw_hud(self, frame: np.ndarray, detections_count: int, latency_ms: float) -> None:
        cv2.rectangle(frame, (10, 10), (520, 96), (18, 18, 18), -1)
        cv2.putText(frame, "YOLOv8 Real-Time Detection", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"FPS: {self._fps:.1f} | Objects: {detections_count} | Inference: {latency_ms:.1f} ms",
            (18, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Device: {self.service.detector.device_name} | Model: {self.service.detector.model_name}",
            (18, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 255, 180),
            1,
            cv2.LINE_AA,
        )

    def _update_fps(self) -> None:
        self._frame_counter += 1
        now = time.perf_counter()
        elapsed = now - self._fps_start
        if elapsed >= 1.0:
            self._fps = self._frame_counter / elapsed
            self._frame_counter = 0
            self._fps_start = now
