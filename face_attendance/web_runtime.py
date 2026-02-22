import base64
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from typing import Deque, Dict, List, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np

from .attendance_service import AttendanceService
from .camera_capture import open_camera_capture
from .camera_discovery import CameraProbeResult, discover_cameras, pick_camera_index
from .config import (
    ANALYTICS_MAX_SIDE,
    ATTENDANCE_COOLDOWN_SECONDS,
    CAMERA_AUTOSELECT,
    CAMERA_HOTPLUG_SCAN_SECONDS,
    CAMERA_PREFER_HIGHEST_INDEX,
    CAMERA_SCAN_MAX_INDEX,
    DUPLICATE_FACE_SIMILARITY_THRESHOLD,
    ENABLE_SKELETON,
    ENABLE_OBJECT_DETECTION,
    ENABLE_OCR,
    FACE_INFERENCE_SCALE,
    FACE_INFERENCE_STRIDE,
    FRAME_FPS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    JPEG_QUALITY,
    OBJECT_CONFIDENCE,
    OBJECT_DETECTION_INTERVAL,
    OBJECT_IMAGE_SIZE,
    OBJECT_IOU,
    OBJECT_MAX_DETECTIONS,
    OBJECT_MODEL,
    OCR_INTERVAL,
    OCR_LANGUAGES,
    OCR_MIN_CONFIDENCE,
    POSE_INFERENCE_SCALE,
    POSE_INFERENCE_STRIDE,
    RECOGNITION_MARGIN,
    RECOGNITION_THRESHOLD,
    SINGLE_USER_RECOGNITION_THRESHOLD,
    TARGET_LOOP_FPS,
    UNKNOWN_CLUSTER_SIMILARITY_THRESHOLD,
)
from .database import AttendanceDatabase, EmployeeRecord
from .exceptions import AttendanceError, CameraError
from .face_engine import FaceEngine
from .logger import setup_logger
from .object_detector import YoloV8Detector
from .object_types import ObjectDetection
from .ocr_engine import EasyOCREngine
from .ocr_preprocessor import FramePreprocessor
from .ocr_types import OCRDetection

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - runtime dependency guard
    mp = None


MAX_HOLOGRAM_FACE_POINTS = 468
TRACK_CONFIRM_FRAMES = 3
TRACK_MAX_CENTER_DISTANCE = 170.0
TRACK_MAX_AGE_SECONDS = 1.2
TRACK_EMBEDDING_ALPHA = 0.35
TRACK_BOX_ALPHA = 0.45
POSE_TRAIL_HISTORY = 14


@dataclass
class PendingUnknown:
    token: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    preview_b64: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class FaceTrack:
    track_id: int
    box: np.ndarray
    center: Tuple[float, float]
    embedding: np.ndarray
    last_seen: float
    candidate_employee_id: Optional[str] = None
    candidate_name: str = "Unknown"
    candidate_hits: int = 0
    unknown_hits: int = 0
    confirmed_employee_id: Optional[str] = None
    confirmed_name: str = "Unknown"
    confirmed_score: float = 0.0


class WebVisionRuntime:
    def __init__(
        self,
        db: AttendanceDatabase,
        engine: FaceEngine,
        threshold: float = RECOGNITION_THRESHOLD,
        camera_index: int = 0,
    ):
        self.enable_skeleton = bool(ENABLE_SKELETON)
        if self.enable_skeleton and mp is None:
            raise AttendanceError(
                "mediapipe is required for hologram skeleton tracking. Install requirements first."
            )

        self.db = db
        self.engine = engine
        self.attendance_service = AttendanceService(db)
        self.threshold = threshold
        self.camera_index = camera_index
        self.logger = setup_logger(self.__class__.__name__)
        self.camera_scan_max_index = CAMERA_SCAN_MAX_INDEX
        self.camera_autoselect = CAMERA_AUTOSELECT
        self.camera_prefer_highest_index = CAMERA_PREFER_HIGHEST_INDEX
        self.manual_camera_pin = False
        self.available_cameras: List[CameraProbeResult] = []
        self.camera_inventory_last_scan_ts = 0.0
        self.capture_lock = threading.Lock()
        self.read_fail_streak = 0
        self.read_fail_error_threshold = 4
        self.read_fail_recover_threshold = 18
        self.min_frame_stale_before_recover_seconds = 1.2
        self.camera_backend_name: Optional[str] = None
        self.active_capture_width = FRAME_WIDTH
        self.active_capture_height = FRAME_HEIGHT
        self.active_capture_fps = FRAME_FPS
        self.last_recover_attempt_ts = 0.0
        self.recover_cooldown_seconds = 3.0
        self.last_hotplug_scan_ts = 0.0
        self.hotplug_scan_seconds = max(3.0, float(CAMERA_HOTPLUG_SCAN_SECONDS))
        self.target_loop_fps = max(0, int(TARGET_LOOP_FPS))
        self.frame_sleep_seconds = (1.0 / self.target_loop_fps) if self.target_loop_fps > 0 else 0.0
        self.jpeg_quality = int(np.clip(JPEG_QUALITY, 45, 95))
        self.face_inference_stride = max(1, int(FACE_INFERENCE_STRIDE))
        self.face_inference_scale = float(np.clip(FACE_INFERENCE_SCALE, 0.4, 1.0))
        self.pose_inference_scale = float(np.clip(POSE_INFERENCE_SCALE, 0.4, 1.0))
        self.analytics_max_side = max(320, int(ANALYTICS_MAX_SIDE))
        self.cached_face_embeddings: List[np.ndarray] = []
        self.cached_face_boxes: List[np.ndarray] = []
        self.cached_face_ts = 0.0

        self.cap: Optional[cv2.VideoCapture] = None
        self.worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.last_camera_jpeg: Optional[bytes] = None
        self.last_skeleton_jpeg: Optional[bytes] = None
        self.pending_unknown: Optional[PendingUnknown] = None
        self.last_seen: Dict[str, float] = {}
        self.latest_scores: Dict[str, float] = {}
        self.latest_fps: float = 0.0
        self.latest_unknown_count: int = 0
        self.latest_tracked_count: int = 0
        self.latest_face_points: int = 0
        self.latest_pose_locked: bool = False
        self.latest_object_count: int = 0
        self.latest_text_count: int = 0
        self.latest_object_latency_ms: float = 0.0
        self.latest_ocr_latency_ms: float = 0.0
        self.last_frame_ts: float = 0.0
        self.last_error: Optional[str] = None
        self.frame_index = 0
        self.last_pose_landmarks = None
        self.last_face_mesh_landmarks = []
        self.last_face_mesh_points: List[Tuple[int, int]] = []
        self.last_object_detections: List[ObjectDetection] = []
        self.last_ocr_detections: List[OCRDetection] = []

        self.next_track_id = 1
        self.face_tracks: Dict[int, FaceTrack] = {}
        self.pose_trail_indexes = (0, 11, 12, 23, 24, 15, 16)
        self.pose_trails: Dict[int, Deque[Tuple[int, int]]] = {
            idx: deque(maxlen=POSE_TRAIL_HISTORY) for idx in self.pose_trail_indexes
        }

        self.employee_ids: List[str] = []
        self.employee_names: List[str] = []
        self.known_matrix: Optional[np.ndarray] = None

        self.current_day = date.today().isoformat()
        self.marked_today = self.db.today_marked_employee_ids()
        self.spoken_today: set[str] = set()
        self.pending_announcements: List[str] = []
        self.last_attempt_times: Dict[str, float] = {}

        self.mp_pose = None
        self.mp_face_mesh = None
        is_cuda = getattr(self.engine.device, "type", "cpu") == "cuda"
        if not is_cuda:
            self.face_inference_stride = max(self.face_inference_stride, 2)
        base_pose_stride = max(1, int(POSE_INFERENCE_STRIDE))
        self.pose_inference_stride = base_pose_stride if is_cuda else max(base_pose_stride, 2)
        self.pose = None
        self.face_mesh = None
        self.face_mesh_connections = []
        if self.enable_skeleton:
            self.mp_pose = mp.solutions.pose
            self.mp_face_mesh = mp.solutions.face_mesh
            pose_model_complexity = 2 if is_cuda else 1
            max_mesh_faces = 5 if is_cuda else 3
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=pose_model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_mesh_faces,
                refine_landmarks=True,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55,
            )
            self.face_mesh_connections = list(self.mp_face_mesh.FACEMESH_TESSELATION)
        self.object_detection_enabled = bool(ENABLE_OBJECT_DETECTION)
        self.object_detection_interval = max(1, int(OBJECT_DETECTION_INTERVAL))
        self.object_detection_img_size = max(256, int(OBJECT_IMAGE_SIZE))
        self.object_detection_max_det = max(20, int(OBJECT_MAX_DETECTIONS))
        self.object_detector: Optional[YoloV8Detector] = None
        if self.object_detection_enabled:
            try:
                self.object_detector = YoloV8Detector(
                    model_path=OBJECT_MODEL,
                    prefer_gpu=is_cuda,
                    conf_threshold=float(np.clip(OBJECT_CONFIDENCE, 0.05, 0.95)),
                    iou_threshold=float(np.clip(OBJECT_IOU, 0.1, 0.95)),
                    img_size=self.object_detection_img_size,
                    max_detections=self.object_detection_max_det,
                )
            except Exception as exc:
                self.object_detection_enabled = False
                self.logger.warning("Object detection disabled: %s", exc)

        self.ocr_enabled = bool(ENABLE_OCR)
        self.ocr_interval = max(1, int(OCR_INTERVAL))
        self.ocr_min_confidence = float(np.clip(OCR_MIN_CONFIDENCE, 0.05, 0.95))
        self.ocr_preprocessor: Optional[FramePreprocessor] = None
        self.ocr_engine: Optional[EasyOCREngine] = None
        if self.ocr_enabled:
            try:
                self.ocr_preprocessor = FramePreprocessor()
                self.ocr_engine = EasyOCREngine(
                    languages=OCR_LANGUAGES,
                    prefer_gpu=is_cuda,
                    min_confidence=self.ocr_min_confidence,
                )
            except Exception as exc:
                self.ocr_enabled = False
                self.logger.warning("OCR disabled: %s", exc)

        self._refresh_known_faces()

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return

        self._refresh_camera_inventory()
        preferred_index = self._pick_start_camera_index(self.camera_index)

        open_errors: list[str] = []
        opened = False
        for index in self._camera_open_candidates(preferred_index):
            try:
                self._open_capture(index)
                opened = True
                break
            except CameraError as exc:
                open_errors.append(str(exc))
                self.logger.warning("Camera source %s failed during startup: %s", index, exc)

        if not opened:
            details = " | ".join(open_errors) if open_errors else "No camera source opened."
            raise CameraError(f"Unable to start runtime camera stream. {details}")

        self.stop_event.clear()
        self.worker = threading.Thread(target=self._loop, name="web-vision-runtime", daemon=True)
        self.worker.start()
        self.logger.info("Web vision runtime started on camera index %s", self.camera_index)

    def stop(self) -> None:
        self.stop_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=3.0)
        self.worker = None

        with self.capture_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

        if self.pose is not None:
            self.pose.close()
        if self.face_mesh is not None:
            self.face_mesh.close()
        self.logger.info("Web vision runtime stopped")

    def list_cameras(self, refresh: bool = False) -> dict:
        now = time.time()
        should_refresh = refresh or not self.available_cameras or (now - self.camera_inventory_last_scan_ts > 25.0)
        if should_refresh:
            self._refresh_camera_inventory()
        return {
            "active_camera_index": self.camera_index,
            "auto_select": self.camera_autoselect and not self.manual_camera_pin,
            "manual_camera_pin": self.manual_camera_pin,
            "cameras": [
                {
                    "index": item.index,
                    "width": item.width,
                    "height": item.height,
                    "fps": round(item.fps, 1),
                    "backend": item.backend,
                }
                for item in self.available_cameras
            ],
        }

    def set_camera_auto_select(self, enabled: bool) -> dict:
        self.camera_autoselect = bool(enabled)
        if self.camera_autoselect:
            self.manual_camera_pin = False
            self._refresh_camera_inventory()
            selected = self._pick_auto_camera_index()
            self._switch_capture(selected)
        return self.list_cameras()

    def switch_camera(self, camera_index: int, pin_manual: bool = True) -> dict:
        self._switch_capture(camera_index)
        if pin_manual:
            self.manual_camera_pin = True
        return self.list_cameras()

    def _refresh_camera_inventory(self) -> None:
        self.available_cameras = discover_cameras(max_index=self.camera_scan_max_index)
        self.camera_inventory_last_scan_ts = time.time()

    def _pick_start_camera_index(self, preferred_index: int) -> int:
        if not self.available_cameras:
            raise CameraError(
                f"No camera sources found. Tried indexes 0..{self.camera_scan_max_index}."
            )
        return pick_camera_index(
            preferred_index=preferred_index,
            available=self.available_cameras,
            prefer_highest_index=self.camera_prefer_highest_index,
        )

    def _pick_auto_camera_index(self) -> int:
        if not self.available_cameras:
            raise CameraError(
                f"No camera sources found. Tried indexes 0..{self.camera_scan_max_index}."
            )
        indices = [entry.index for entry in self.available_cameras]
        if self.camera_prefer_highest_index:
            return max(indices)
        return min(indices)

    def _camera_open_candidates(self, preferred_index: int) -> list[int]:
        candidates = [preferred_index]
        for entry in self.available_cameras:
            if entry.index not in candidates:
                candidates.append(entry.index)
        return candidates

    def _open_capture(self, camera_index: int) -> None:
        cap, backend_name = open_camera_capture(camera_index)
        cv2.setUseOptimized(True)
        width, height, fps = self._configure_capture_profile(cap, camera_index, backend_name)

        with self.capture_lock:
            if self.cap is not None:
                self.cap.release()
            self.cap = cap

        self.camera_index = camera_index
        self.camera_backend_name = backend_name
        self.active_capture_width = width
        self.active_capture_height = height
        self.active_capture_fps = fps
        self.read_fail_streak = 0
        self.logger.info(
            "Camera stream set to index %s via %s backend (%sx%s @ %s FPS)",
            camera_index,
            backend_name,
            width,
            height,
            fps,
        )

    @staticmethod
    def _capture_profiles() -> list[tuple[int, int, int]]:
        raw_profiles = [
            (FRAME_WIDTH, FRAME_HEIGHT, FRAME_FPS),
            (960, 540, min(FRAME_FPS, 24)),
            (640, 480, min(FRAME_FPS, 20)),
        ]
        unique: list[tuple[int, int, int]] = []
        for profile in raw_profiles:
            if profile not in unique:
                unique.append(profile)
        return unique

    def _configure_capture_profile(
        self,
        cap: cv2.VideoCapture,
        camera_index: int,
        backend_name: str,
    ) -> tuple[int, int, int]:
        width, height, fps = self._capture_profiles()[0]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
        actual_fps = int(round(cap.get(cv2.CAP_PROP_FPS) or fps))
        if actual_fps <= 0:
            actual_fps = fps
        self.logger.info(
            "Camera %s via %s configured to %sx%s @ %s FPS",
            camera_index,
            backend_name,
            actual_width,
            actual_height,
            actual_fps,
        )
        return actual_width, actual_height, actual_fps

    def _switch_capture(self, camera_index: int) -> None:
        if camera_index == self.camera_index and self.cap is not None:
            return
        self._open_capture(camera_index)

    def _attempt_autorecover_camera(self) -> None:
        self._refresh_camera_inventory()
        if not self.available_cameras:
            return

        # If a manually pinned camera is still present, keep it.
        if self.manual_camera_pin:
            pinned_present = any(entry.index == self.camera_index for entry in self.available_cameras)
            if pinned_present:
                return
            # Pinned source disappeared (for example mobile camera disconnected).
            # Fall back to auto-select so runtime can recover to laptop webcam.
            self.manual_camera_pin = False
            self.logger.warning(
                "Pinned camera source %s unavailable; switching to available source.",
                self.camera_index,
            )

        if not self.camera_autoselect:
            return

        next_camera = self._pick_auto_camera_index()
        with self.capture_lock:
            cap_missing = self.cap is None
        if next_camera != self.camera_index or cap_missing:
            self._switch_capture(next_camera)

    def refresh_known_faces(self) -> None:
        with self.lock:
            self._refresh_known_faces()
            self.marked_today = self.db.today_marked_employee_ids()

    def get_jpeg_frame(self, stream_name: str) -> Optional[bytes]:
        with self.lock:
            if stream_name == "camera":
                return self.last_camera_jpeg
            if stream_name == "skeleton":
                return self.last_skeleton_jpeg
            return None

    def get_dashboard_state(self) -> dict:
        now = time.time()
        with self.lock:
            recognized = []
            for name, seen_ts in self.last_seen.items():
                if now - seen_ts <= 2.5:
                    recognized.append(
                        {
                            "name": name,
                            "score": round(self.latest_scores.get(name, 0.0), 3),
                            "last_seen_ms": int((now - seen_ts) * 1000),
                        }
                    )

            pending = None
            if self.pending_unknown and now - self.pending_unknown.updated_at <= 8.0:
                pending = {
                    "token": self.pending_unknown.token,
                    "sample_count": len(self.pending_unknown.embeddings),
                    "preview_b64": self.pending_unknown.preview_b64,
                }

            announcements = list(self.pending_announcements)
            self.pending_announcements.clear()
            frame_age_ms: Optional[int] = None
            camera_live = False
            if self.last_frame_ts > 0.0:
                frame_age = max(0.0, now - self.last_frame_ts)
                frame_age_ms = int(frame_age * 1000)
                camera_live = frame_age <= 2.0 and self.last_error is None

            return {
                "recognized": sorted(recognized, key=lambda x: x["score"], reverse=True),
                "known_count": len(self.employee_ids),
                "fps": round(self.latest_fps, 1),
                "target_fps": self.target_loop_fps,
                "pending_unknown": pending,
                "announcements": announcements,
                "face_points": self.latest_face_points,
                "unknown_count": self.latest_unknown_count,
                "tracked_count": self.latest_tracked_count,
                "pose_locked": self.latest_pose_locked,
                "object_count": self.latest_object_count,
                "text_count": self.latest_text_count,
                "object_latency_ms": round(self.latest_object_latency_ms, 1),
                "ocr_latency_ms": round(self.latest_ocr_latency_ms, 1),
                "object_detection_enabled": self.object_detection_enabled,
                "ocr_enabled": self.ocr_enabled,
                "skeleton_enabled": self.enable_skeleton,
                "camera_index": self.camera_index,
                "camera_backend": self.camera_backend_name,
                "camera_profile": {
                    "width": self.active_capture_width,
                    "height": self.active_capture_height,
                    "fps": self.active_capture_fps,
                },
                "camera_auto_select": self.camera_autoselect and not self.manual_camera_pin,
                "camera_live": camera_live,
                "camera_frame_age_ms": frame_age_ms,
                "error": self.last_error,
            }

    def clear_pending_unknown(self) -> None:
        with self.lock:
            self.pending_unknown = None

    def register_pending_unknown(self, employee_id: str, name: str) -> dict:
        employee_id = employee_id.strip()
        name = name.strip()
        if not employee_id:
            raise AttendanceError("Employee ID is required.")
        if not name:
            raise AttendanceError("Name is required.")

        with self.lock:
            pending = self.pending_unknown
            if pending is None:
                embeddings: List[np.ndarray] = []
            else:
                embeddings = [emb.copy() for emb in pending.embeddings]

        if len(embeddings) < 5:
            raise AttendanceError("Need at least 5 stable unknown face samples before saving.")

        matrix = np.vstack(embeddings).astype(np.float32)
        vector = matrix.mean(axis=0)
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            raise AttendanceError("Failed to normalize pending face embedding.")
        vector = vector / norm

        duplicate_name = None
        duplicate_id = None
        duplicate_score = 0.0
        with self.lock:
            if self.known_matrix is not None and self.known_matrix.size:
                scores = self.known_matrix @ vector
                idx = int(np.argmax(scores))
                duplicate_score = float(scores[idx])
                duplicate_name = self.employee_names[idx]
                duplicate_id = self.employee_ids[idx]

        if (
            duplicate_id is not None
            and duplicate_id != employee_id
            and duplicate_score >= DUPLICATE_FACE_SIMILARITY_THRESHOLD
        ):
            raise AttendanceError(
                f"Captured face is too similar to existing user '{duplicate_name}' ({duplicate_id}). "
                "Try again with a different person."
            )

        self.db.upsert_employee(employee_id=employee_id, name=name, encoding=vector)
        self.refresh_known_faces()
        self.clear_pending_unknown()
        self.logger.info("New user saved from pending unknown: %s (%s)", name, employee_id)
        return {"employee_id": employee_id, "name": name}

    def _refresh_known_faces(self) -> None:
        records: List[EmployeeRecord] = self.db.list_employees()
        if not records:
            self.employee_ids = []
            self.employee_names = []
            self.known_matrix = None
            return

        self.employee_ids = [record.employee_id for record in records]
        self.employee_names = [record.name for record in records]
        matrix = np.vstack([record.encoding for record in records]).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.known_matrix = matrix / np.clip(norms, 1e-9, None)

    def _loop(self) -> None:
        prev_time = time.perf_counter()
        while not self.stop_event.is_set():
            loop_started = time.perf_counter()
            with self.capture_lock:
                if self.cap is None:
                    ok = False
                    frame = None
                else:
                    ok, frame = self.cap.read()
            if self.cap is None:
                now = time.time()
                if (now - self.last_recover_attempt_ts) >= self.recover_cooldown_seconds:
                    self.last_recover_attempt_ts = now
                    self._recover_capture_after_read_failures()
                time.sleep(0.08)
                continue
            if not ok or frame is None:
                self.read_fail_streak += 1
                if self.read_fail_streak >= self.read_fail_error_threshold:
                    with self.lock:
                        self.last_error = (
                            f"Camera frame read failed on source {self.camera_index}. "
                            "Check USB/mobile camera connection."
                        )
                if self.read_fail_streak >= self.read_fail_recover_threshold:
                    now = time.time()
                    if self.last_frame_ts > 0.0:
                        frame_stale_seconds = now - self.last_frame_ts
                    else:
                        frame_stale_seconds = float("inf")
                    if (
                        frame_stale_seconds >= self.min_frame_stale_before_recover_seconds
                        and (now - self.last_recover_attempt_ts) >= self.recover_cooldown_seconds
                    ):
                        self.last_recover_attempt_ts = now
                        self._recover_capture_after_read_failures()
                        self.read_fail_streak = 0
                time.sleep(0.03)
                continue
            self.read_fail_streak = 0
            self._auto_switch_on_hotplug_if_needed()

            self._rollover_day_if_needed()

            now = time.perf_counter()
            delta = max(1e-6, now - prev_time)
            prev_time = now
            fps = 1.0 / delta

            self.frame_index += 1
            self._process_frame(frame, fps)

            if self.frame_sleep_seconds > 0.0:
                elapsed = time.perf_counter() - loop_started
                remaining = self.frame_sleep_seconds - elapsed
                if remaining > 0.0:
                    time.sleep(min(remaining, 0.03))

    def _auto_switch_on_hotplug_if_needed(self) -> None:
        if self.manual_camera_pin:
            return
        if not self.camera_autoselect:
            return

        now = time.time()
        if (now - self.last_hotplug_scan_ts) < self.hotplug_scan_seconds:
            return
        self.last_hotplug_scan_ts = now

        # Avoid camera inventory probing while stream is already unstable.
        with self.lock:
            stream_healthy = self.last_error is None and (
                self.last_frame_ts > 0.0 and (now - self.last_frame_ts) <= 2.0
            )
        if not stream_healthy:
            return

        try:
            # Probe other camera indices only; probing the active source can disrupt
            # some USB/mobile webcam drivers and trigger false reconnect loops.
            scanned = discover_cameras(
                max_index=self.camera_scan_max_index,
                exclude_indices={self.camera_index},
            )
            with self.capture_lock:
                cap_ready = self.cap is not None
            if cap_ready:
                scanned.append(
                    CameraProbeResult(
                        index=self.camera_index,
                        width=self.active_capture_width,
                        height=self.active_capture_height,
                        fps=float(self.active_capture_fps),
                        backend=self.camera_backend_name or "active",
                    )
                )
            dedup: dict[int, CameraProbeResult] = {}
            for item in scanned:
                if item.index not in dedup:
                    dedup[item.index] = item
            self.available_cameras = sorted(dedup.values(), key=lambda item: item.index)
            self.camera_inventory_last_scan_ts = now
            if not self.available_cameras:
                return
            preferred = self._pick_auto_camera_index()
            with self.capture_lock:
                cap_missing = self.cap is None
            if preferred != self.camera_index or cap_missing:
                previous = self.camera_index
                self._switch_capture(preferred)
                if preferred != previous:
                    self.logger.info(
                        "Auto-switched camera from %s to %s after hot-plug scan.",
                        previous,
                        preferred,
                    )
        except CameraError:
            return
        except Exception as exc:
            self.logger.warning("Hot-plug camera scan failed: %s", exc)

    def _recover_capture_after_read_failures(self) -> None:
        current_index = self.camera_index

        # Force release first so stale handles do not block re-open.
        with self.capture_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

        try:
            self._open_capture(current_index)
            self.logger.warning("Recovered camera stream by reopening source %s.", current_index)
            return
        except Exception as exc:
            self.logger.warning(
                "Failed to reopen camera source %s after read failures: %s",
                current_index,
                exc,
            )

        self._attempt_autorecover_camera()
        with self.lock:
            self.last_error = (
                f"Camera source {current_index} is unstable. "
                "Try another source from Camera Source -> Select source."
            )

    def _process_frame(self, frame: np.ndarray, fps: float) -> None:
        try:
            camera_frame = frame.copy()
            if self.enable_skeleton:
                skeleton_frame = self._render_skeleton_frame(frame)
                face_point_count = len(self.last_face_mesh_points)
                pose_locked = self.last_pose_landmarks is not None
            else:
                skeleton_frame = camera_frame.copy()
                face_point_count = 0
                pose_locked = True

            embeddings, boxes = self._extract_face_detections(frame)
            detections = list(zip(embeddings, boxes))
            active_tracks = self._assign_tracks(detections)

            frame_seen: Dict[str, float] = {}
            frame_known_ids: Dict[str, str] = {}
            unknown_count = 0
            for track in active_tracks:
                employee_id, label, score, status = self._classify_track(track)
                if status == "known":
                    prev = frame_seen.get(label)
                    if prev is None or score > prev:
                        frame_seen[label] = score
                        frame_known_ids[label] = employee_id
                elif status == "unknown":
                    unknown_count += 1
                    self._capture_unknown_candidate(track.embedding, frame, track.box)

                self._draw_face_box(
                    camera_frame,
                    track.box,
                    label,
                    score,
                    status=status,
                    track_id=track.track_id,
                )

            for known_name, employee_id in frame_known_ids.items():
                self._handle_known_person(employee_id=employee_id, name=known_name)

            object_count = self.latest_object_count if self.object_detection_enabled else 0
            text_count = self.latest_text_count if self.ocr_enabled else 0
            object_latency_ms = self.latest_object_latency_ms if self.object_detection_enabled else 0.0
            ocr_latency_ms = self.latest_ocr_latency_ms if self.ocr_enabled else 0.0

            if self.object_detection_enabled and self.object_detector is not None:
                object_count, object_latency_ms = self._run_object_detection(frame, camera_frame)
            if self.ocr_enabled and self.ocr_engine is not None and self.ocr_preprocessor is not None:
                text_count, ocr_latency_ms = self._run_ocr_detection(frame, camera_frame)

            self._draw_camera_hud(
                camera_frame,
                recognized=frame_seen,
                unknown_count=unknown_count,
                tracked_count=len(active_tracks),
                fps=fps,
                object_count=object_count,
                text_count=text_count,
                object_latency_ms=object_latency_ms,
                ocr_latency_ms=ocr_latency_ms,
                object_enabled=self.object_detection_enabled,
                ocr_enabled=self.ocr_enabled,
            )
            self._draw_skeleton_hud(
                skeleton_frame,
                fps=fps,
                face_points=face_point_count,
                pose_locked=pose_locked,
            )

            ok_cam, encoded_cam = cv2.imencode(
                ".jpg",
                camera_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            ok_skel, encoded_skel = cv2.imencode(
                ".jpg",
                skeleton_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok_cam or not ok_skel:
                return

            now = time.time()
            with self.lock:
                self.last_camera_jpeg = encoded_cam.tobytes()
                self.last_skeleton_jpeg = encoded_skel.tobytes()
                for name, score in frame_seen.items():
                    self.last_seen[name] = now
                    self.latest_scores[name] = score
                self.latest_fps = fps
                self.latest_unknown_count = unknown_count
                self.latest_tracked_count = len(active_tracks)
                self.latest_face_points = face_point_count
                self.latest_pose_locked = pose_locked
                self.latest_object_count = object_count
                self.latest_text_count = text_count
                self.latest_object_latency_ms = object_latency_ms
                self.latest_ocr_latency_ms = ocr_latency_ms
                self.last_frame_ts = now
                self.last_error = None
        except Exception as exc:
            self.logger.exception("Frame processing failed")
            fallback = frame.copy()
            cv2.rectangle(fallback, (0, 0), (fallback.shape[1], 68), (25, 25, 25), -1)
            cv2.putText(
                fallback,
                "AI pipeline degraded - camera feed kept live",
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                fallback,
                "Recovery in progress...",
                (14, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            ok_cam, encoded_cam = cv2.imencode(
                ".jpg",
                fallback,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            now = time.time()
            with self.lock:
                if ok_cam:
                    self.last_camera_jpeg = encoded_cam.tobytes()
                self.last_frame_ts = now
                self.latest_fps = fps
                self.last_error = str(exc)

    def _extract_face_detections(self, frame: np.ndarray) -> tuple[List[np.ndarray], List[np.ndarray]]:
        now = time.time()
        infer_now = (self.frame_index % self.face_inference_stride) == 0 or not self.cached_face_embeddings

        if not infer_now and (now - self.cached_face_ts) <= 1.0:
            return (
                [item.copy() for item in self.cached_face_embeddings],
                [item.copy() for item in self.cached_face_boxes],
            )

        inference_frame, scale = self._resize_for_inference(frame, self.face_inference_scale)
        batch = self.engine.extract_embeddings(inference_frame)

        h, w = frame.shape[:2]
        scaled_boxes: List[np.ndarray] = []
        inv_scale = 1.0 / max(1e-6, scale)
        for box in batch.boxes:
            projected = box.astype(np.float32).copy()
            projected[0::2] *= inv_scale
            projected[1::2] *= inv_scale
            projected[0::2] = np.clip(projected[0::2], 0, max(0, w - 1))
            projected[1::2] = np.clip(projected[1::2], 0, max(0, h - 1))
            scaled_boxes.append(projected)

        self.cached_face_embeddings = [item.astype(np.float32, copy=True) for item in batch.embeddings]
        self.cached_face_boxes = [item.copy() for item in scaled_boxes]
        self.cached_face_ts = now
        return self.cached_face_embeddings, self.cached_face_boxes

    def _run_object_detection(
        self,
        source_frame: np.ndarray,
        camera_frame: np.ndarray,
    ) -> tuple[int, float]:
        if self.object_detector is None:
            return 0, 0.0

        if (self.frame_index % self.object_detection_interval) == 0 or not self.last_object_detections:
            inference_frame, scale = self._resize_for_inference(source_frame, 1.0)
            try:
                start = time.perf_counter()
                detections = self.object_detector.detect(inference_frame)
                latency_ms = (time.perf_counter() - start) * 1000.0
                if scale < 0.999:
                    detections = self._scale_object_detections(
                        detections,
                        inverse_scale=(1.0 / scale),
                        source=source_frame,
                    )
                self.last_object_detections = detections
            except Exception as exc:
                self.object_detection_enabled = False
                self.last_object_detections = []
                self.logger.warning("Object detection disabled after runtime failure: %s", exc)
                return 0, 0.0
        else:
            latency_ms = self.latest_object_latency_ms

        for detection in self.last_object_detections[:24]:
            x1, y1, x2, y2 = detection.bbox
            color = self._color_for_label(detection.label)
            cv2.rectangle(camera_frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.putText(
                camera_frame,
                f"{detection.label} {detection.confidence:.2f}",
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

        return len(self.last_object_detections), float(latency_ms)

    def _run_ocr_detection(
        self,
        source_frame: np.ndarray,
        camera_frame: np.ndarray,
    ) -> tuple[int, float]:
        if self.ocr_engine is None or self.ocr_preprocessor is None:
            return 0, 0.0

        if (self.frame_index % self.ocr_interval) == 0 or not self.last_ocr_detections:
            inference_frame, scale = self._resize_for_inference(source_frame, 1.0)
            try:
                enhanced = self.ocr_preprocessor.enhance(inference_frame)
                start = time.perf_counter()
                detections = self.ocr_engine.detect_and_recognize(enhanced)
                latency_ms = (time.perf_counter() - start) * 1000.0
                if scale < 0.999:
                    detections = self._scale_ocr_detections(
                        detections,
                        inverse_scale=(1.0 / scale),
                        source=source_frame,
                    )
                self.last_ocr_detections = detections
            except Exception as exc:
                self.ocr_enabled = False
                self.last_ocr_detections = []
                self.logger.warning("OCR disabled after runtime failure: %s", exc)
                return 0, 0.0
        else:
            latency_ms = self.latest_ocr_latency_ms

        for detection in self.last_ocr_detections[:14]:
            polygon = np.array(detection.polygon, dtype=np.int32)
            if polygon.ndim != 2 or polygon.shape[0] < 4:
                continue
            cv2.polylines(camera_frame, [polygon], True, (75, 230, 255), 2, cv2.LINE_AA)
            x = int(np.min(polygon[:, 0]))
            y = int(np.min(polygon[:, 1]))
            cv2.putText(
                camera_frame,
                detection.text[:42],
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (75, 230, 255),
                1,
                cv2.LINE_AA,
            )

        return len(self.last_ocr_detections), float(latency_ms)

    def _resize_for_inference(self, frame: np.ndarray, requested_scale: float) -> tuple[np.ndarray, float]:
        height, width = frame.shape[:2]
        max_side = max(height, width)
        scale = float(np.clip(requested_scale, 0.2, 1.0))
        max_side_limit = self.analytics_max_side
        if max_side > max_side_limit:
            scale = min(scale, max_side_limit / float(max_side))
        if scale >= 0.999:
            return frame, 1.0

        resized = cv2.resize(
            frame,
            (max(2, int(width * scale)), max(2, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    @staticmethod
    def _scale_object_detections(
        detections: List[ObjectDetection],
        inverse_scale: float,
        source: np.ndarray,
    ) -> List[ObjectDetection]:
        h, w = source.shape[:2]
        scaled: List[ObjectDetection] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            sx1 = int(np.clip(round(x1 * inverse_scale), 0, max(0, w - 1)))
            sy1 = int(np.clip(round(y1 * inverse_scale), 0, max(0, h - 1)))
            sx2 = int(np.clip(round(x2 * inverse_scale), 0, max(0, w - 1)))
            sy2 = int(np.clip(round(y2 * inverse_scale), 0, max(0, h - 1)))
            scaled.append(
                ObjectDetection(
                    class_id=det.class_id,
                    label=det.label,
                    confidence=det.confidence,
                    bbox=(sx1, sy1, sx2, sy2),
                )
            )
        return scaled

    @staticmethod
    def _scale_ocr_detections(
        detections: List[OCRDetection],
        inverse_scale: float,
        source: np.ndarray,
    ) -> List[OCRDetection]:
        h, w = source.shape[:2]
        scaled: List[OCRDetection] = []
        for det in detections:
            polygon: List[Tuple[int, int]] = []
            for x, y in det.polygon:
                sx = int(np.clip(round(x * inverse_scale), 0, max(0, w - 1)))
                sy = int(np.clip(round(y * inverse_scale), 0, max(0, h - 1)))
                polygon.append((sx, sy))
            scaled.append(
                OCRDetection(
                    polygon=polygon,
                    text=det.text,
                    confidence=det.confidence,
                    angle_degrees=det.angle_degrees,
                )
            )
        return scaled

    @staticmethod
    def _color_for_label(label: str) -> Tuple[int, int, int]:
        seed = (sum(ord(ch) for ch in label) + 1) * 17
        return (
            int((seed * 13) % 255),
            int((seed * 23) % 255),
            int((seed * 31) % 255),
        )

    def _assign_tracks(self, detections: List[Tuple[np.ndarray, np.ndarray]]) -> List[FaceTrack]:
        now = time.time()
        stale_ids = [tid for tid, track in self.face_tracks.items() if now - track.last_seen > TRACK_MAX_AGE_SECONDS]
        for tid in stale_ids:
            del self.face_tracks[tid]

        active_tracks: List[FaceTrack] = []
        unmatched_track_ids = set(self.face_tracks.keys())
        for embedding, box in detections:
            box = box.astype(np.float32)
            center = self._box_center(box)
            match_track_id = self._find_best_track_match(box=box, center=center, candidate_ids=unmatched_track_ids)
            embedding_norm = self._normalize_embedding(embedding)

            if match_track_id is None:
                track = FaceTrack(
                    track_id=self.next_track_id,
                    box=box.copy(),
                    center=center,
                    embedding=embedding_norm,
                    last_seen=now,
                )
                self.face_tracks[track.track_id] = track
                self.next_track_id += 1
            else:
                track = self.face_tracks[match_track_id]
                track.box = (TRACK_BOX_ALPHA * track.box) + ((1.0 - TRACK_BOX_ALPHA) * box)
                track.center = self._box_center(track.box)
                blended = ((1.0 - TRACK_EMBEDDING_ALPHA) * track.embedding) + (TRACK_EMBEDDING_ALPHA * embedding_norm)
                track.embedding = self._normalize_embedding(blended)
                track.last_seen = now
                unmatched_track_ids.discard(match_track_id)

            active_tracks.append(track)

        # Keep very recent unmatched tracks to avoid visual flicker on skipped inference frames.
        for track_id in list(unmatched_track_ids):
            track = self.face_tracks.get(track_id)
            if track is None:
                continue
            if (now - track.last_seen) <= 0.35:
                active_tracks.append(track)

        return active_tracks

    def _find_best_track_match(
        self,
        box: np.ndarray,
        center: Tuple[float, float],
        candidate_ids: set[int],
    ) -> Optional[int]:
        if not candidate_ids:
            return None

        box_w = float(max(1.0, box[2] - box[0]))
        box_h = float(max(1.0, box[3] - box[1]))
        scale_limit = 0.65 * float(np.hypot(box_w, box_h))
        distance_limit = max(TRACK_MAX_CENTER_DISTANCE, scale_limit)

        best_track_id = None
        best_distance = float("inf")
        cx, cy = center
        for track_id in candidate_ids:
            track = self.face_tracks[track_id]
            tx, ty = track.center
            distance = float(np.hypot(cx - tx, cy - ty))
            if distance <= distance_limit and distance < best_distance:
                best_distance = distance
                best_track_id = track_id

        return best_track_id

    def _classify_track(self, track: FaceTrack) -> Tuple[Optional[str], str, float, str]:
        employee_id, name, score, _margin = self._match_face(track.embedding)
        if employee_id is not None:
            if track.candidate_employee_id == employee_id:
                track.candidate_hits += 1
            else:
                track.candidate_employee_id = employee_id
                track.candidate_name = name
                track.candidate_hits = 1

            track.unknown_hits = 0
            if track.candidate_hits >= TRACK_CONFIRM_FRAMES:
                track.confirmed_employee_id = employee_id
                track.confirmed_name = name
                track.confirmed_score = score
                return employee_id, name, score, "known"

            return None, f"Locking {name}", score, "locking"

        track.candidate_employee_id = None
        track.candidate_name = "Unknown"
        track.candidate_hits = 0
        track.unknown_hits += 1

        if track.unknown_hits >= 2:
            track.confirmed_employee_id = None
            track.confirmed_name = "Unknown"
            track.confirmed_score = 0.0

        return None, "Unknown", score, "unknown"

    def _handle_known_person(self, employee_id: str, name: str) -> None:
        now = time.time()
        with self.lock:
            last_try = self.last_attempt_times.get(employee_id, 0.0)
            if now - last_try < ATTENDANCE_COOLDOWN_SECONDS:
                return
            self.last_attempt_times[employee_id] = now

        if employee_id not in self.marked_today:
            inserted = self.attendance_service.mark_now(employee_id)
            if inserted:
                with self.lock:
                    self.marked_today.add(employee_id)
                self.logger.info("Attendance marked from web runtime for %s (%s)", name, employee_id)

        with self.lock:
            if employee_id not in self.spoken_today:
                self.spoken_today.add(employee_id)
                self.pending_announcements.append(f"{name} recognized")
                self.pending_announcements = self.pending_announcements[-20:]

    def _rollover_day_if_needed(self) -> None:
        today = date.today().isoformat()
        if today == self.current_day:
            return

        with self.lock:
            self.current_day = today
            self.marked_today = self.db.today_marked_employee_ids()
            self.spoken_today.clear()
            self.last_attempt_times.clear()
            self.pending_announcements.clear()
        self.logger.info("Date rollover detected. Attendance cache reset for %s", today)

    def _render_skeleton_frame(self, source_frame: np.ndarray) -> np.ndarray:
        if not self.enable_skeleton or self.pose is None or self.face_mesh is None:
            return source_frame.copy()

        stride = self.pose_inference_stride
        if self.latest_fps > 0 and self.latest_fps < 9.0:
            stride = max(stride, 2)

        if self.frame_index % stride == 0:
            pose_frame, _ = self._resize_for_inference(source_frame, self.pose_inference_scale)
            rgb = cv2.cvtColor(pose_frame, cv2.COLOR_BGR2RGB)
            pose_result = self.pose.process(rgb)
            mesh_result = self.face_mesh.process(rgb)

            self.last_pose_landmarks = pose_result.pose_landmarks if pose_result else None
            self.last_face_mesh_landmarks = mesh_result.multi_face_landmarks if mesh_result else []
            self.last_face_mesh_points = self._extract_face_points(
                mesh_result,
                source_frame.shape[1],
                source_frame.shape[0],
            )
            self._update_pose_trails(source_frame.shape[1], source_frame.shape[0])

        h, w = source_frame.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (8, 14, 22)

        self._draw_hologram_grid(canvas)
        self._draw_pose_trails(canvas)
        self._draw_pose_overlay(canvas)
        self._draw_face_mesh_overlay(canvas)
        self._draw_face_points_overlay(canvas)
        return canvas

    def _update_pose_trails(self, width: int, height: int) -> None:
        if not self.last_pose_landmarks:
            return

        landmarks = self.last_pose_landmarks.landmark
        for idx in self.pose_trail_indexes:
            landmark = landmarks[idx]
            if landmark.visibility < 0.35:
                continue
            x = int(np.clip(landmark.x, 0.0, 1.0) * width)
            y = int(np.clip(landmark.y, 0.0, 1.0) * height)
            self.pose_trails[idx].append((x, y))

    @staticmethod
    def _extract_face_points(mesh_result, width: int, height: int) -> List[Tuple[int, int]]:
        if mesh_result is None or not mesh_result.multi_face_landmarks:
            return []

        points: List[Tuple[int, int]] = []
        for face_landmarks in mesh_result.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = int(np.clip(landmark.x, 0.0, 1.0) * width)
                y = int(np.clip(landmark.y, 0.0, 1.0) * height)
                points.append((x, y))
                if len(points) >= MAX_HOLOGRAM_FACE_POINTS:
                    return points
        return points

    @staticmethod
    def _draw_hologram_grid(canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        step = 46
        for x in range(0, w, step):
            cv2.line(canvas, (x, 0), (x, h), (18, 40, 58), 1, cv2.LINE_AA)
        for y in range(0, h, step):
            cv2.line(canvas, (0, y), (w, y), (18, 40, 58), 1, cv2.LINE_AA)

        horizon = int(h * 0.68)
        cv2.line(canvas, (0, horizon), (w, horizon), (32, 78, 102), 1, cv2.LINE_AA)

    def _draw_pose_trails(self, canvas: np.ndarray) -> None:
        overlay = np.zeros_like(canvas)
        for idx in self.pose_trail_indexes:
            trail = list(self.pose_trails[idx])
            if len(trail) < 2:
                continue
            for i in range(1, len(trail)):
                fade = i / len(trail)
                color = (int(70 + 80 * fade), int(160 + 60 * fade), int(250 - 20 * fade))
                thickness = 1 if i < len(trail) - 1 else 2
                cv2.line(overlay, trail[i - 1], trail[i], color, thickness, cv2.LINE_AA)

        glow = cv2.GaussianBlur(overlay, (0, 0), sigmaX=3.4, sigmaY=3.4)
        canvas[:] = cv2.addWeighted(canvas, 1.0, glow, 0.8, 0)
        canvas[:] = cv2.addWeighted(canvas, 1.0, overlay, 0.86, 0)

    def _draw_pose_overlay(self, canvas: np.ndarray) -> None:
        if not self.last_pose_landmarks:
            cv2.putText(
                canvas,
                "POSE SIGNAL: SEARCHING",
                (30, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.86,
                (100, 220, 255),
                2,
                cv2.LINE_AA,
            )
            return

        h, w = canvas.shape[:2]
        overlay = np.zeros_like(canvas)
        landmarks = self.last_pose_landmarks.landmark
        for start_idx, end_idx in self.mp_pose.POSE_CONNECTIONS:
            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]
            vis = min(p1.visibility, p2.visibility)
            if vis < 0.35:
                continue

            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)
            color = (int(90 + vis * 80), int(210 + vis * 45), 255)
            thickness = 2 if vis >= 0.6 else 1
            cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        for landmark in landmarks:
            if landmark.visibility < 0.35:
                continue
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            radius = 3 if landmark.visibility > 0.6 else 2
            cv2.circle(overlay, (cx, cy), radius, (210, 255, 135), -1, cv2.LINE_AA)

        glow = cv2.GaussianBlur(overlay, (0, 0), sigmaX=4.8, sigmaY=4.8)
        canvas[:] = cv2.addWeighted(canvas, 1.0, glow, 0.78, 0)
        canvas[:] = cv2.addWeighted(canvas, 1.0, overlay, 0.92, 0)

    def _draw_face_mesh_overlay(self, canvas: np.ndarray) -> None:
        if not self.last_face_mesh_landmarks:
            return

        h, w = canvas.shape[:2]
        overlay = np.zeros_like(canvas)
        for face_landmarks in self.last_face_mesh_landmarks[:2]:
            lm = face_landmarks.landmark
            for idx, (start_idx, end_idx) in enumerate(self.face_mesh_connections):
                if idx % 2:
                    continue

                p1 = lm[start_idx]
                p2 = lm[end_idx]
                x1, y1 = int(np.clip(p1.x, 0.0, 1.0) * w), int(np.clip(p1.y, 0.0, 1.0) * h)
                x2, y2 = int(np.clip(p2.x, 0.0, 1.0) * w), int(np.clip(p2.y, 0.0, 1.0) * h)
                cv2.line(overlay, (x1, y1), (x2, y2), (82, 225, 255), 1, cv2.LINE_AA)

        glow = cv2.GaussianBlur(overlay, (0, 0), sigmaX=2.8, sigmaY=2.8)
        canvas[:] = cv2.addWeighted(canvas, 1.0, glow, 0.82, 0)
        canvas[:] = cv2.addWeighted(canvas, 1.0, overlay, 0.62, 0)

    def _draw_face_points_overlay(self, canvas: np.ndarray) -> None:
        if not self.last_face_mesh_points:
            return

        overlay = np.zeros_like(canvas)
        pulse_phase = self.frame_index % 8
        for idx, (x, y) in enumerate(self.last_face_mesh_points):
            radius = 1 if idx % 3 else 2
            if (idx + pulse_phase) % 11 == 0:
                radius += 1
            color = (88, 255, 244) if idx % 2 == 0 else (130, 236, 255)
            cv2.circle(overlay, (x, y), radius, color, -1, cv2.LINE_AA)

        glow = cv2.GaussianBlur(overlay, (0, 0), sigmaX=3.2, sigmaY=3.2)
        canvas[:] = cv2.addWeighted(canvas, 1.0, glow, 0.86, 0)
        canvas[:] = cv2.addWeighted(canvas, 1.0, overlay, 0.95, 0)

    def _match_face(self, query: np.ndarray) -> Tuple[Optional[str], str, float, float]:
        if self.known_matrix is None or self.known_matrix.size == 0:
            return None, "Unknown", 0.0, 0.0

        query = self._normalize_embedding(query)
        scores = self.known_matrix @ query
        idx = int(np.argmax(scores))
        best = float(scores[idx])

        if scores.size == 1:
            required = max(self.threshold + 0.08, SINGLE_USER_RECOGNITION_THRESHOLD)
            if best >= required:
                return self.employee_ids[idx], self.employee_names[idx], best, best
            return None, "Unknown", best, 0.0

        if scores.size >= 2:
            second = float(np.partition(scores, -2)[-2])
        else:
            second = -1.0
        margin = best - second

        if best >= self.threshold and margin >= RECOGNITION_MARGIN:
            return self.employee_ids[idx], self.employee_names[idx], best, margin
        return None, "Unknown", best, margin

    def _capture_unknown_candidate(self, embedding: np.ndarray, frame: np.ndarray, box: np.ndarray) -> None:
        now = time.time()
        normalized = self._normalize_embedding(embedding)

        with self.lock:
            pending = self.pending_unknown
            if pending and now - pending.updated_at > 5.0:
                pending = None
                self.pending_unknown = None

            if pending is None:
                pending = PendingUnknown(token=uuid4().hex)
                self.pending_unknown = pending
            else:
                avg = np.mean(np.vstack(pending.embeddings), axis=0) if pending.embeddings else normalized
                avg = self._normalize_embedding(avg.astype(np.float32))
                similarity = float(avg @ normalized)
                if similarity < UNKNOWN_CLUSTER_SIMILARITY_THRESHOLD:
                    pending = PendingUnknown(token=uuid4().hex)
                    self.pending_unknown = pending

            if len(pending.embeddings) < 24:
                pending.embeddings.append(normalized)

            pending.updated_at = now
            pending.preview_b64 = self._face_crop_to_base64(frame, box)

    @staticmethod
    def _face_crop_to_base64(frame: np.ndarray, box: np.ndarray) -> str:
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * 0.25)
        pad_y = int((y2 - y1) * 0.25)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return ""
        ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return ""
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    @staticmethod
    def _draw_face_box(
        frame: np.ndarray,
        box: np.ndarray,
        label: str,
        score: float,
        status: str,
        track_id: int,
    ) -> None:
        x1, y1, x2, y2 = [int(v) for v in box]
        if status == "known":
            color = (72, 255, 228)
        elif status == "locking":
            color = (60, 220, 255)
        else:
            color = (20, 150, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"T{track_id} {label} {score:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_camera_hud(
        frame: np.ndarray,
        recognized: Dict[str, float],
        unknown_count: int,
        tracked_count: int,
        fps: float,
        object_count: int,
        text_count: int,
        object_latency_ms: float,
        ocr_latency_ms: float,
        object_enabled: bool,
        ocr_enabled: bool,
    ) -> None:
        _, w = frame.shape[:2]
        panel_h = 108
        cv2.rectangle(frame, (0, 0), (w, panel_h), (10, 20, 30), -1)
        text = (
            f"LIVE CAMERA RECOGNITION | FPS: {fps:.1f} | "
            f"Known: {len(recognized)} | Unknown: {unknown_count} | Tracks: {tracked_count}"
        )
        cv2.putText(
            frame,
            text,
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (240, 250, 255),
            2,
            cv2.LINE_AA,
        )
        object_status = f"Objects: {object_count}" if object_enabled else "Objects: off"
        ocr_status = f"Text: {text_count}" if ocr_enabled else "Text: off"
        cv2.putText(
            frame,
            f"{object_status} ({object_latency_ms:.1f} ms) | {ocr_status} ({ocr_latency_ms:.1f} ms)",
            (18, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (170, 235, 245),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_skeleton_hud(frame: np.ndarray, fps: float, face_points: int, pose_locked: bool) -> None:
        _, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 78), (6, 15, 22), -1)
        pose_status = "LOCKED" if pose_locked else "SEARCHING"
        cv2.putText(
            frame,
            f"SKELETON HOLOGRAM | FACE NODES: {face_points} | POSE: {pose_status} | FPS: {fps:.1f}",
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (175, 244, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _box_center(box: np.ndarray) -> Tuple[float, float]:
        return float((box[0] + box[2]) * 0.5), float((box[1] + box[3]) * 0.5)

    @staticmethod
    def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
        normalized = vector.astype(np.float32, copy=True)
        norm = float(np.linalg.norm(normalized))
        if norm <= 1e-9:
            return normalized
        return normalized / norm
