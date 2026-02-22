from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class VisionAISettings:
    project_root: Path
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    frame_fps: int = 30
    queue_size: int = 3
    face_match_threshold: float = 0.55
    face_registration_samples: int = 25
    yolo_model: str = "yolov8s.pt"
    yolo_confidence: float = 0.25
    yolo_iou: float = 0.45
    yolo_img_size: int = 640
    yolo_max_detections: int = 300
    yolo_ensemble_enabled: bool = True
    yolo_ensemble_img_size: int = 960
    yolo_tta: bool = True
    yolo_temporal_seconds: float = 0.65
    ocr_languages: Sequence[str] = ("en",)
    ocr_confidence: float = 0.35
    ocr_multi_pass: bool = True
    ocr_upscale_factor: float = 1.5
    ocr_temporal_seconds: float = 1.8
    ocr_max_entries: int = 90
    db_url: str = "sqlite:///vision_ai/database/vision_ai.db"
    ocr_snapshot_path: Path = Path("vision_ai/logs/ocr_snapshots.txt")
    benchmark_seconds: int = 0

    @property
    def use_gpu(self) -> bool:
        return torch.cuda.is_available()

    @property
    def db_path(self) -> Path:
        if self.db_url.startswith("sqlite:///"):
            rel = self.db_url.replace("sqlite:///", "", 1)
            return self.project_root / rel
        return self.project_root / "vision_ai/database/vision_ai.db"

    @property
    def log_dir(self) -> Path:
        return self.project_root / "vision_ai/logs"

    @property
    def model_dir(self) -> Path:
        return self.project_root / "vision_ai/models"

    @property
    def config_dir(self) -> Path:
        return self.project_root / "vision_ai/config"

    @property
    def embedding_key_path(self) -> Path:
        return self.config_dir / ".embedding.key"

    def ensure_directories(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ocr_snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, project_root: Path) -> "VisionAISettings":
        return cls(
            project_root=project_root,
            camera_index=_env_int("VISION_CAMERA_INDEX", 0),
            frame_width=_env_int("VISION_FRAME_WIDTH", 1280),
            frame_height=_env_int("VISION_FRAME_HEIGHT", 720),
            frame_fps=_env_int("VISION_FRAME_FPS", 30),
            queue_size=max(2, _env_int("VISION_QUEUE_SIZE", 3)),
            face_match_threshold=_env_float("VISION_FACE_THRESHOLD", 0.55),
            face_registration_samples=max(20, min(30, _env_int("VISION_REGISTRATION_SAMPLES", 25))),
            yolo_model=os.getenv("VISION_YOLO_MODEL", "yolov8s.pt"),
            yolo_confidence=_env_float("VISION_YOLO_CONF", 0.25),
            yolo_iou=_env_float("VISION_YOLO_IOU", 0.45),
            yolo_img_size=_env_int("VISION_YOLO_IMGSZ", 640),
            yolo_max_detections=max(50, _env_int("VISION_YOLO_MAX_DET", 300)),
            yolo_ensemble_enabled=_env_bool("VISION_YOLO_ENSEMBLE", True),
            yolo_ensemble_img_size=max(512, _env_int("VISION_YOLO_ENSEMBLE_IMGSZ", 960)),
            yolo_tta=_env_bool("VISION_YOLO_TTA", True),
            yolo_temporal_seconds=max(0.0, _env_float("VISION_YOLO_TEMPORAL_SECONDS", 0.65)),
            ocr_languages=tuple(
                token.strip() for token in os.getenv("VISION_OCR_LANGS", "en").split(",") if token.strip()
            )
            or ("en",),
            ocr_confidence=_env_float("VISION_OCR_CONF", 0.35),
            ocr_multi_pass=_env_bool("VISION_OCR_MULTI_PASS", True),
            ocr_upscale_factor=max(1.0, _env_float("VISION_OCR_UPSCALE", 1.5)),
            ocr_temporal_seconds=max(0.0, _env_float("VISION_OCR_TEMPORAL_SECONDS", 1.8)),
            ocr_max_entries=max(20, _env_int("VISION_OCR_MAX_ENTRIES", 90)),
            db_url=os.getenv("VISION_DB_URL", "sqlite:///vision_ai/database/vision_ai.db"),
            ocr_snapshot_path=Path(os.getenv("VISION_OCR_SNAPSHOT", "vision_ai/logs/ocr_snapshots.txt")),
            benchmark_seconds=max(0, _env_int("VISION_BENCHMARK_SECONDS", 0)),
        )
