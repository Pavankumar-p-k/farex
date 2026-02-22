from __future__ import annotations

import argparse
import shlex
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Callable

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from face_attendance.camera_capture import open_camera_capture
from vision_ai.config import VisionAISettings
from vision_ai.database import VisionAIDatabase
from vision_ai.face_module import ArcFaceEngine, FaceRecognitionService
from vision_ai.object_module import YOLODetector, YOLOTrainer, write_dataset_yaml
from vision_ai.ocr_module import RealTimeOCREngine
from vision_ai.utils import PerformanceTracker, configure_logger, configure_runtime_environment, put_latest
from vision_ai.utils.types import FaceResult, FramePacket, OCRResult, ObjectResult, WorkerEnvelope

configure_runtime_environment()
import cv2  # noqa: E402


class VisionRuntime:
    def __init__(self, settings: VisionAISettings, prefer_gpu: bool = True, enable_console: bool = True):
        self.settings = settings
        self.enable_console = enable_console
        self.settings.ensure_directories()
        self.logger = configure_logger(settings.log_dir)
        self.metrics = PerformanceTracker()
        self.stop_event = threading.Event()
        self.command_queue: Queue[str] = Queue(maxsize=16)

        self.db = VisionAIDatabase(
            db_path=self.settings.db_path,
            key_path=self.settings.embedding_key_path,
        )

        self.face_service = FaceRecognitionService(
            engine=ArcFaceEngine(prefer_gpu=prefer_gpu),
            db=self.db,
            match_threshold=self.settings.face_match_threshold,
        )
        self.object_detector = YOLODetector(
            model_path=self.settings.yolo_model,
            confidence=self.settings.yolo_confidence,
            iou=self.settings.yolo_iou,
            image_size=self.settings.yolo_img_size,
            prefer_gpu=prefer_gpu,
            max_detections=self.settings.yolo_max_detections,
            ensemble_enabled=self.settings.yolo_ensemble_enabled,
            ensemble_img_size=self.settings.yolo_ensemble_img_size,
            tta_enabled=self.settings.yolo_tta,
            temporal_seconds=self.settings.yolo_temporal_seconds,
        )
        self.ocr_engine = RealTimeOCREngine(
            languages=self.settings.ocr_languages,
            confidence=self.settings.ocr_confidence,
            prefer_gpu=prefer_gpu,
            multi_pass=self.settings.ocr_multi_pass,
            upscale_factor=self.settings.ocr_upscale_factor,
            temporal_seconds=self.settings.ocr_temporal_seconds,
            max_entries=self.settings.ocr_max_entries,
        )

        qsize = self.settings.queue_size
        self.face_queue: Queue[FramePacket] = Queue(maxsize=qsize)
        self.object_queue: Queue[FramePacket] = Queue(maxsize=qsize)
        self.ocr_queue: Queue[FramePacket] = Queue(maxsize=qsize)
        self.display_queue: Queue[FramePacket] = Queue(maxsize=qsize)
        self.result_queue: Queue[WorkerEnvelope] = Queue(maxsize=max(8, qsize * 4))

        self.aggregate_lock = threading.Lock()
        self.latest_face: FaceResult | None = None
        self.latest_object: ObjectResult | None = None
        self.latest_ocr: OCRResult | None = None
        self.status_message = ""
        self.status_expiry = 0.0

        self.start_time = time.perf_counter()
        self.displayed_frames = 0

    def run(self) -> dict[str, float]:
        self.logger.info(
            "Runtime initialized",
            extra={
                "event": "runtime_initialized",
                "camera_index": self.settings.camera_index,
                "gpu_enabled": self.settings.use_gpu,
                "face_device": self.face_service.engine.device_name,
                "yolo_device": self.object_detector.device_name,
                "ocr_device": self.ocr_engine.device_name,
                "yolo_ensemble": self.settings.yolo_ensemble_enabled,
                "yolo_tta": self.settings.yolo_tta,
                "ocr_multi_pass": self.settings.ocr_multi_pass,
            },
        )

        if self.settings.benchmark_seconds > 0:
            self.logger.info(
                "Benchmark mode enabled",
                extra={"event": "benchmark", "seconds": self.settings.benchmark_seconds},
            )

        workers = [
            threading.Thread(target=self._camera_worker, name="camera-thread", daemon=True),
            threading.Thread(target=self._face_worker, name="face-thread", daemon=True),
            threading.Thread(target=self._object_worker, name="object-thread", daemon=True),
            threading.Thread(target=self._ocr_worker, name="ocr-thread", daemon=True),
            threading.Thread(target=self._aggregation_worker, name="aggregation-thread", daemon=True),
        ]
        if self.enable_console:
            workers.append(threading.Thread(target=self._command_listener, name="command-thread", daemon=True))

        for worker in workers:
            worker.start()

        try:
            self._display_loop()
        finally:
            self.stop_event.set()
            for worker in workers:
                worker.join(timeout=1.5)
            try:
                cv2.destroyAllWindows()
            except Exception:
                # Headless OpenCV builds may not support HighGUI cleanup calls.
                pass

        duration = max(1e-6, time.perf_counter() - self.start_time)
        snapshot = self.metrics.snapshot()
        return {
            "duration_seconds": duration,
            "display_fps": self.displayed_frames / duration,
            "camera_fps": snapshot.get("camera", {}).get("fps", 0.0),
            "face_fps": snapshot.get("face", {}).get("fps", 0.0),
            "face_latency_ms": snapshot.get("face", {}).get("latency_ms", 0.0),
            "yolo_fps": snapshot.get("yolo", {}).get("fps", 0.0),
            "yolo_latency_ms": snapshot.get("yolo", {}).get("latency_ms", 0.0),
            "ocr_fps": snapshot.get("ocr", {}).get("fps", 0.0),
            "ocr_latency_ms": snapshot.get("ocr", {}).get("latency_ms", 0.0),
        }

    def _camera_worker(self) -> None:
        try:
            cap, backend_name = open_camera_capture(self.settings.camera_index)
        except Exception as exc:
            self.logger.error(
                "Failed to open camera",
                extra={
                    "event": "camera_open_failed",
                    "camera_index": self.settings.camera_index,
                    "error": str(exc),
                },
            )
            self.stop_event.set()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.settings.frame_fps)
        cv2.setUseOptimized(True)
        self.logger.info(
            "Camera stream opened",
            extra={
                "event": "camera_opened",
                "camera_index": self.settings.camera_index,
                "backend": backend_name,
            },
        )

        frame_id = 0
        started = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                frame_id += 1
                packet = FramePacket(
                    frame_id=frame_id,
                    timestamp=time.perf_counter(),
                    frame=frame,
                )

                put_latest(self.face_queue, packet)
                put_latest(self.object_queue, packet)
                put_latest(self.ocr_queue, packet)
                put_latest(self.display_queue, packet)
                self.metrics.update("camera", 0.0)

                if self.settings.benchmark_seconds > 0:
                    if (time.perf_counter() - started) >= self.settings.benchmark_seconds:
                        self.stop_event.set()
                        break
        finally:
            cap.release()

    def _face_worker(self) -> None:
        self._run_worker(
            module="face",
            queue_obj=self.face_queue,
            processor=lambda packet: self.face_service.process_frame(packet.frame_id, packet.frame),
        )

    def _object_worker(self) -> None:
        self._run_worker(
            module="yolo",
            queue_obj=self.object_queue,
            processor=lambda packet: self.object_detector.detect(packet.frame_id, packet.frame),
        )

    def _ocr_worker(self) -> None:
        self._run_worker(
            module="ocr",
            queue_obj=self.ocr_queue,
            processor=lambda packet: self.ocr_engine.detect(packet.frame_id, packet.frame),
        )

    def _run_worker(
        self,
        module: str,
        queue_obj: Queue[FramePacket],
        processor: Callable[[FramePacket], FaceResult | ObjectResult | OCRResult],
    ) -> None:
        while not self.stop_event.is_set():
            try:
                packet = queue_obj.get(timeout=0.2)
            except Empty:
                continue

            started = time.perf_counter()
            try:
                payload = processor(packet)
            except Exception as exc:
                self.logger.exception(
                    "Worker failure",
                    extra={"event": "worker_failure", "module_name": module, "error": str(exc)},
                )
                continue

            latency_ms = (time.perf_counter() - started) * 1000.0
            self.metrics.update(module, latency_ms)
            put_latest(
                self.result_queue,
                WorkerEnvelope(module=module, frame_id=packet.frame_id, payload=payload),
            )

    def _aggregation_worker(self) -> None:
        last_metrics_log = time.perf_counter()
        while not self.stop_event.is_set():
            self._drain_results_once(timeout=0.2)
            self._drain_commands()

            now = time.perf_counter()
            if now - last_metrics_log >= 5.0:
                snapshot = self.metrics.snapshot()
                self.logger.info(
                    "Performance snapshot",
                    extra={"event": "metrics", "snapshot": snapshot},
                )
                last_metrics_log = now

    def _drain_results_once(self, timeout: float = 0.0) -> None:
        try:
            envelope = self.result_queue.get(timeout=timeout)
        except Empty:
            return
        self._store_envelope(envelope)
        while True:
            try:
                envelope = self.result_queue.get_nowait()
            except Empty:
                break
            self._store_envelope(envelope)

    def _store_envelope(self, envelope: WorkerEnvelope) -> None:
        with self.aggregate_lock:
            if envelope.module == "face":
                self.latest_face = envelope.payload
            elif envelope.module == "yolo":
                self.latest_object = envelope.payload
            elif envelope.module == "ocr":
                self.latest_ocr = envelope.payload

    def _command_listener(self) -> None:
        self.logger.info(
            "Console commands ready",
            extra={
                "event": "commands_help",
                "hint": "register <EMP_ID> <Full Name>, summary, export [path], quit",
            },
        )
        while not self.stop_event.is_set():
            try:
                command = input().strip()
            except EOFError:
                break
            if not command:
                continue
            put_latest(self.command_queue, command)

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self.command_queue.get_nowait()
            except Empty:
                return
            self._handle_command(command)

    def _handle_command(self, command: str) -> None:
        parts = shlex.split(command)
        if not parts:
            return
        action = parts[0].lower()

        if action == "register" and len(parts) >= 3:
            employee_id = parts[1]
            name = " ".join(parts[2:])
            message = self.face_service.start_registration(
                employee_id=employee_id,
                name=name,
                samples=self.settings.face_registration_samples,
            )
            self._set_status(message)
            self.logger.info("Registration command", extra={"event": "register", "employee_id": employee_id})
            return

        if action == "summary":
            rows = self.db.daily_attendance_summary()
            self._set_status(f"Today's attendance: {len(rows)} records")
            self.logger.info("Attendance summary", extra={"event": "summary", "count": len(rows)})
            return

        if action == "export":
            output = Path(parts[1]) if len(parts) > 1 else self.settings.log_dir / "attendance_today.csv"
            output = self.db.export_attendance_csv(output_path=output)
            self._set_status(f"Attendance exported: {output}")
            self.logger.info("Attendance exported", extra={"event": "export_csv", "path": str(output)})
            return

        if action in {"quit", "exit"}:
            self._set_status("Stopping Vision AI runtime.")
            self.stop_event.set()
            return

        self._set_status("Unknown command. Use: register, summary, export, quit")

    def _display_loop(self) -> None:
        window_name = "Vision AI Platform | Q quit | S save OCR | commands in terminal"
        headless_display = False
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except Exception:
            headless_display = True
            self.logger.warning(
                "Display window unavailable. Running in headless mode.",
                extra={"event": "headless_display"},
            )

        while not self.stop_event.is_set():
            try:
                packet = self.display_queue.get(timeout=0.2)
            except Empty:
                self._drain_results_once(timeout=0.0)
                continue

            self._drain_results_once(timeout=0.0)
            frame = packet.frame.copy()
            with self.aggregate_lock:
                face_result = self.latest_face
                object_result = self.latest_object
                ocr_result = self.latest_ocr

            if face_result and abs(packet.frame_id - face_result.frame_id) <= 30:
                self._draw_face_overlay(frame, face_result)
            if object_result and abs(packet.frame_id - object_result.frame_id) <= 30:
                self._draw_object_overlay(frame, object_result)
            if ocr_result and abs(packet.frame_id - ocr_result.frame_id) <= 45:
                self._draw_ocr_overlay(frame, ocr_result)

            self._draw_metrics_overlay(frame)
            self._draw_status(frame)

            self.displayed_frames += 1

            if not headless_display:
                try:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        self.stop_event.set()
                        break
                    if key == ord("s"):
                        self._save_ocr_snapshot(ocr_result)
                except Exception:
                    headless_display = True
                    self.logger.warning(
                        "Display loop switched to headless mode.",
                        extra={"event": "headless_display_runtime"},
                    )

        if not headless_display:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

    def _draw_face_overlay(self, frame, face_result: FaceResult) -> None:
        for identity in face_result.identities:
            x1, y1, x2, y2 = identity.bbox
            known = identity.employee_id != "UNKNOWN"
            color = (0, 220, 80) if known else (0, 140, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            tag = f"{identity.name} {identity.confidence:.2f}"
            if identity.new_attendance:
                tag += " [ATTENDED]"
            cv2.putText(
                frame,
                tag,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        if face_result.registration_mode:
            progress = int(face_result.registration_progress * 100)
            cv2.putText(
                frame,
                f"Registration: {progress}%",
                (20, frame.shape[0] - 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def _draw_object_overlay(self, frame, object_result: ObjectResult) -> None:
        for detection in object_result.detections:
            x1, y1, x2, y2 = detection.bbox
            color = self._label_color(detection.label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{detection.label} {detection.confidence:.2f}"
            cv2.putText(
                frame,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    def _draw_ocr_overlay(self, frame, ocr_result: OCRResult) -> None:
        for entry in ocr_result.entries:
            polygon = entry.polygon
            if len(polygon) < 4:
                continue
            points = cv2.convexHull(np.array(polygon, dtype=np.int32))
            cv2.polylines(frame, [points], True, (255, 220, 60), 2, cv2.LINE_AA)
            x, y = polygon[0]
            cv2.putText(
                frame,
                f"{entry.text} ({entry.confidence:.2f})",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 220, 60),
                2,
                cv2.LINE_AA,
            )

    def _draw_metrics_overlay(self, frame) -> None:
        snapshot = self.metrics.snapshot()
        cv2.rectangle(frame, (10, 10), (620, 150), (20, 20, 20), -1)
        cv2.putText(frame, "Vision AI Platform", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        camera = snapshot.get("camera", {})
        face = snapshot.get("face", {})
        yolo = snapshot.get("yolo", {})
        ocr = snapshot.get("ocr", {})

        cv2.putText(
            frame,
            f"Camera FPS: {camera.get('fps', 0.0):.1f} | Display FPS: {self._display_fps():.1f}",
            (18, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            (
                f"Face FPS {face.get('fps', 0.0):.1f} ({face.get('latency_ms', 0.0):.1f} ms) | "
                f"YOLO FPS {yolo.get('fps', 0.0):.1f} ({yolo.get('latency_ms', 0.0):.1f} ms)"
            ),
            (18, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 255, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"OCR FPS {ocr.get('fps', 0.0):.1f} ({ocr.get('latency_ms', 0.0):.1f} ms)",
            (18, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 255, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            (
                f"Face:{self.face_service.engine.device_name} "
                f"YOLO:{self.object_detector.device_name} "
                f"OCR:{self.ocr_engine.device_name}"
            ),
            (18, 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 220, 150),
            1,
            cv2.LINE_AA,
        )

    def _draw_status(self, frame) -> None:
        if self.status_message and time.time() < self.status_expiry:
            cv2.putText(
                frame,
                self.status_message,
                (18, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def _save_ocr_snapshot(self, ocr_result: OCRResult | None) -> None:
        if ocr_result is None:
            self._set_status("No OCR snapshot available.")
            return

        unique_lines: list[str] = []
        seen: set[str] = set()
        for item in ocr_result.entries:
            norm = item.text.strip().lower()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            unique_lines.append(item.text.strip())

        with self.settings.ocr_snapshot_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{datetime.now().isoformat(timespec='seconds')}]\n")
            if unique_lines:
                for line in unique_lines:
                    handle.write(f"{line}\n")
            else:
                handle.write("(empty)\n")
            handle.write("\n")

        self.db.save_ocr_snapshot(unique_lines, source="live_camera")
        self._set_status(f"Saved OCR snapshot ({len(unique_lines)} lines)")

    def _set_status(self, message: str, ttl_seconds: float = 2.5) -> None:
        self.status_message = message
        self.status_expiry = time.time() + ttl_seconds

    def _display_fps(self) -> float:
        duration = max(1e-6, time.perf_counter() - self.start_time)
        return self.displayed_frames / duration

    @staticmethod
    def _label_color(label: str) -> tuple[int, int, int]:
        seed = sum(ord(ch) for ch in label) + 1
        return (
            int((seed * 31) % 255),
            int((seed * 57) % 255),
            int((seed * 97) % 255),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Integrated real-time Vision AI Platform")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run integrated face + object + OCR platform")
    run.add_argument("--camera", type=int, default=None, help="Camera index")
    run.add_argument("--width", type=int, default=None, help="Frame width")
    run.add_argument("--height", type=int, default=None, help="Frame height")
    run.add_argument("--fps", type=int, default=None, help="Camera FPS")
    run.add_argument("--queue-size", type=int, default=None, help="Per-module frame queue size")
    run.add_argument("--face-threshold", type=float, default=None, help="Face recognition threshold")
    run.add_argument("--register-samples", type=int, default=None, help="Live registration target samples")
    run.add_argument("--yolo-model", default=None, help="YOLOv8 model weights")
    run.add_argument("--yolo-conf", type=float, default=None, help="YOLO confidence threshold")
    run.add_argument("--yolo-iou", type=float, default=None, help="YOLO IoU threshold")
    run.add_argument("--yolo-imgsz", type=int, default=None, help="YOLO image size")
    run.add_argument("--yolo-max-det", type=int, default=None, help="YOLO max detections per frame")
    run.add_argument("--yolo-ensemble-imgsz", type=int, default=None, help="Second YOLO pass image size")
    run.add_argument("--disable-yolo-ensemble", action="store_true", help="Disable YOLO multi-scale ensemble")
    run.add_argument("--disable-yolo-tta", action="store_true", help="Disable YOLO test-time augmentation")
    run.add_argument("--yolo-temporal-seconds", type=float, default=None, help="YOLO temporal smoothing window")
    run.add_argument("--ocr-langs", nargs="+", default=None, help="OCR languages (example: en hi)")
    run.add_argument("--ocr-conf", type=float, default=None, help="OCR confidence threshold")
    run.add_argument("--disable-ocr-multi-pass", action="store_true", help="Disable OCR multi-pass reads")
    run.add_argument("--ocr-upscale", type=float, default=None, help="OCR upscale factor for small text")
    run.add_argument("--ocr-temporal-seconds", type=float, default=None, help="OCR temporal stabilization window")
    run.add_argument("--ocr-max-entries", type=int, default=None, help="Max OCR lines kept per frame")
    run.add_argument(
        "--accuracy-mode",
        choices=["balanced", "max"],
        default=None,
        help="Apply tuned accuracy preset before explicit overrides",
    )
    run.add_argument("--db-url", default=None, help="Database URL (sqlite:///...)")
    run.add_argument("--benchmark-seconds", type=int, default=None, help="Benchmark duration")
    run.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    run.add_argument("--no-console", action="store_true", help="Disable terminal command listener")

    init_data = subparsers.add_parser("init-object-data", help="Generate YOLO dataset YAML")
    init_data.add_argument("--output", type=Path, required=True)
    init_data.add_argument("--train", required=True)
    init_data.add_argument("--val", required=True)
    init_data.add_argument("--test", default=None)
    init_data.add_argument("--names", nargs="+", required=True)

    train = subparsers.add_parser("train-objects", help="Train a custom YOLOv8 model")
    train.add_argument("--data", type=Path, required=True, help="Dataset YAML")
    train.add_argument("--model", default="yolov8n.pt", help="Base model path")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--project", type=Path, default=Path("vision_ai/models/training_runs"))
    train.add_argument("--name", default="custom")
    train.add_argument("--no-gpu", action="store_true")

    export = subparsers.add_parser("export-attendance", help="Export attendance CSV for a day")
    export.add_argument("--day", default=None, help="Date as YYYY-MM-DD; default today")
    export.add_argument("--output", type=Path, default=Path("vision_ai/logs/attendance_export.csv"))
    return parser


def _apply_overrides(settings: VisionAISettings, args: argparse.Namespace) -> VisionAISettings:
    if args.accuracy_mode == "max":
        settings.yolo_confidence = min(settings.yolo_confidence, 0.22)
        settings.yolo_iou = max(settings.yolo_iou, 0.50)
        settings.yolo_img_size = max(settings.yolo_img_size, 960)
        settings.yolo_ensemble_enabled = True
        settings.yolo_ensemble_img_size = max(settings.yolo_ensemble_img_size, 1280)
        settings.yolo_tta = True
        settings.yolo_temporal_seconds = max(settings.yolo_temporal_seconds, 1.2)
        settings.ocr_multi_pass = True
        settings.ocr_upscale_factor = max(settings.ocr_upscale_factor, 1.8)
        settings.ocr_temporal_seconds = max(settings.ocr_temporal_seconds, 2.4)
        settings.ocr_max_entries = max(settings.ocr_max_entries, 120)
    elif args.accuracy_mode == "balanced":
        settings.yolo_ensemble_enabled = True
        settings.yolo_tta = True
        settings.ocr_multi_pass = True

    if args.camera is not None:
        settings.camera_index = args.camera
    if args.width is not None:
        settings.frame_width = args.width
    if args.height is not None:
        settings.frame_height = args.height
    if args.fps is not None:
        settings.frame_fps = args.fps
    if args.queue_size is not None:
        settings.queue_size = max(2, args.queue_size)
    if args.face_threshold is not None:
        settings.face_match_threshold = args.face_threshold
    if args.register_samples is not None:
        settings.face_registration_samples = max(20, min(30, args.register_samples))
    if args.yolo_model is not None:
        settings.yolo_model = args.yolo_model
    if args.yolo_conf is not None:
        settings.yolo_confidence = args.yolo_conf
    if args.yolo_iou is not None:
        settings.yolo_iou = args.yolo_iou
    if args.yolo_imgsz is not None:
        settings.yolo_img_size = args.yolo_imgsz
    if args.yolo_max_det is not None:
        settings.yolo_max_detections = max(20, args.yolo_max_det)
    if args.yolo_ensemble_imgsz is not None:
        settings.yolo_ensemble_img_size = max(512, args.yolo_ensemble_imgsz)
    if args.disable_yolo_ensemble:
        settings.yolo_ensemble_enabled = False
    if args.disable_yolo_tta:
        settings.yolo_tta = False
    if args.yolo_temporal_seconds is not None:
        settings.yolo_temporal_seconds = max(0.0, args.yolo_temporal_seconds)
    if args.ocr_langs is not None:
        settings.ocr_languages = tuple(args.ocr_langs)
    if args.ocr_conf is not None:
        settings.ocr_confidence = args.ocr_conf
    if args.disable_ocr_multi_pass:
        settings.ocr_multi_pass = False
    if args.ocr_upscale is not None:
        settings.ocr_upscale_factor = max(1.0, args.ocr_upscale)
    if args.ocr_temporal_seconds is not None:
        settings.ocr_temporal_seconds = max(0.0, args.ocr_temporal_seconds)
    if args.ocr_max_entries is not None:
        settings.ocr_max_entries = max(20, args.ocr_max_entries)
    if args.db_url is not None:
        settings.db_url = args.db_url
    if args.benchmark_seconds is not None:
        settings.benchmark_seconds = max(0, args.benchmark_seconds)
    return settings


def _run_command(args: argparse.Namespace, project_root: Path) -> int:
    settings = VisionAISettings.from_env(project_root=project_root)
    settings = _apply_overrides(settings, args)
    runtime = VisionRuntime(
        settings=settings,
        prefer_gpu=not args.no_gpu,
        enable_console=not args.no_console,
    )
    summary = runtime.run()
    print(
        "Benchmark summary: "
        f"duration={summary['duration_seconds']:.2f}s, "
        f"display_fps={summary['display_fps']:.2f}, "
        f"face={summary['face_fps']:.2f}fps/{summary['face_latency_ms']:.1f}ms, "
        f"yolo={summary['yolo_fps']:.2f}fps/{summary['yolo_latency_ms']:.1f}ms, "
        f"ocr={summary['ocr_fps']:.2f}fps/{summary['ocr_latency_ms']:.1f}ms"
    )
    return 0


def _init_object_data_command(args: argparse.Namespace) -> int:
    output = write_dataset_yaml(
        output_path=args.output,
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        class_names=args.names,
    )
    print(f"Created dataset YAML: {output}")
    return 0


def _train_objects_command(args: argparse.Namespace) -> int:
    trainer = YOLOTrainer(prefer_gpu=not args.no_gpu)
    weights = trainer.train(
        data_yaml=args.data,
        model_path=args.model,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch_size=args.batch,
        project=args.project,
        run_name=args.name,
    )
    print(f"Training complete. Weights: {weights}")
    return 0


def _export_attendance_command(args: argparse.Namespace, project_root: Path) -> int:
    settings = VisionAISettings.from_env(project_root=project_root)
    settings.ensure_directories()
    db = VisionAIDatabase(db_path=settings.db_path, key_path=settings.embedding_key_path)
    output = db.export_attendance_csv(output_path=args.output, day=args.day)
    print(f"Attendance CSV exported: {output}")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    try:
        if args.command == "run":
            return _run_command(args, project_root)
        if args.command == "init-object-data":
            return _init_object_data_command(args)
        if args.command == "train-objects":
            return _train_objects_command(args)
        if args.command == "export-attendance":
            return _export_attendance_command(args, project_root)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 1
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
