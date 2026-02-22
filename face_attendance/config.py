import os
from pathlib import Path

import torch


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    values = tuple(token.strip() for token in raw.split(",") if token.strip())
    return values or default


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
DB_PATH = DATA_DIR / "attendance.db"

# Webcam settings
CAMERA_INDEX = _int_env("FACE_CAMERA_INDEX", 0)
FRAME_WIDTH = _int_env("FACE_FRAME_WIDTH", 1280)
FRAME_HEIGHT = _int_env("FACE_FRAME_HEIGHT", 720)
FRAME_FPS = _int_env("FACE_FRAME_FPS", 30)
CAMERA_SCAN_MAX_INDEX = _int_env("FACE_CAMERA_SCAN_MAX_INDEX", 8)
CAMERA_AUTOSELECT = _bool_env("FACE_CAMERA_AUTOSELECT", True)
CAMERA_PREFER_HIGHEST_INDEX = _bool_env("FACE_CAMERA_PREFER_HIGHEST_INDEX", True)
CAMERA_HOTPLUG_SCAN_SECONDS = _int_env("FACE_CAMERA_HOTPLUG_SCAN_SECONDS", 8)
TARGET_LOOP_FPS = _int_env("FACE_TARGET_LOOP_FPS", 20)
JPEG_QUALITY = _int_env("FACE_JPEG_QUALITY", 78)
FACE_INFERENCE_STRIDE = _int_env("FACE_FACE_INFERENCE_STRIDE", 2)
FACE_INFERENCE_SCALE = _float_env("FACE_FACE_INFERENCE_SCALE", 0.85)
POSE_INFERENCE_STRIDE = _int_env("FACE_POSE_INFERENCE_STRIDE", 2)
POSE_INFERENCE_SCALE = _float_env("FACE_POSE_INFERENCE_SCALE", 0.75)
ANALYTICS_MAX_SIDE = _int_env("FACE_ANALYTICS_MAX_SIDE", 960)
ENABLE_SKELETON = _bool_env("FACE_ENABLE_SKELETON", True)

# Optional scene intelligence overlays for camera stream.
ENABLE_OBJECT_DETECTION = _bool_env("FACE_ENABLE_OBJECT_DETECTION", False)
OBJECT_DETECTION_INTERVAL = _int_env("FACE_OBJECT_DETECTION_INTERVAL", 5)
OBJECT_MODEL = os.getenv("FACE_OBJECT_MODEL", "yolov8n.pt")
OBJECT_CONFIDENCE = _float_env("FACE_OBJECT_CONFIDENCE", 0.30)
OBJECT_IOU = _float_env("FACE_OBJECT_IOU", 0.45)
OBJECT_IMAGE_SIZE = _int_env("FACE_OBJECT_IMAGE_SIZE", 512)
OBJECT_MAX_DETECTIONS = _int_env("FACE_OBJECT_MAX_DETECTIONS", 100)

ENABLE_OCR = _bool_env("FACE_ENABLE_OCR", False)
OCR_INTERVAL = _int_env("FACE_OCR_INTERVAL", 8)
OCR_MIN_CONFIDENCE = _float_env("FACE_OCR_MIN_CONFIDENCE", 0.40)
OCR_LANGUAGES = _csv_env("FACE_OCR_LANGUAGES", ("en",))

# Registration settings
REGISTRATION_SAMPLES = 20
SAMPLE_EVERY_N_FRAMES = 4
FACE_DETECTION_THRESHOLD = 0.94
MIN_FACE_SIZE = 80

# Recognition settings
RECOGNITION_THRESHOLD = 0.82
RECOGNITION_MARGIN = 0.08
SINGLE_USER_RECOGNITION_THRESHOLD = 0.90
DUPLICATE_FACE_SIMILARITY_THRESHOLD = 0.88
UNKNOWN_CLUSTER_SIMILARITY_THRESHOLD = 0.80
ATTENDANCE_COOLDOWN_SECONDS = 5

# Runtime settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
