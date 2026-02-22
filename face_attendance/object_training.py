from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from .exceptions import TrainingError

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - handled at runtime.
    YOLO = None


def write_dataset_yaml(
    output_path: Path,
    train_path: str,
    val_path: str,
    class_names: Sequence[str],
    test_path: str | None = None,
) -> Path:
    if not class_names:
        raise TrainingError("At least one class name is required.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"train: {train_path}",
        f"val: {val_path}",
    ]
    if test_path:
        lines.append(f"test: {test_path}")
    lines.append("")
    lines.append("names:")
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


class YOLOTrainingService:
    def __init__(self, prefer_gpu: bool = True) -> None:
        if YOLO is None:
            raise TrainingError(
                "Ultralytics YOLO is not installed. Install dependencies with: pip install -r requirements.txt"
            )
        self.device = "cuda:0" if (prefer_gpu and torch.cuda.is_available()) else "cpu"

    def train(
        self,
        data_yaml: Path,
        model_path: str = "yolov8n.pt",
        epochs: int = 50,
        image_size: int = 640,
        batch_size: int = 16,
        project_dir: Path = Path("runs/object_training"),
        run_name: str = "custom",
    ) -> Path:
        if not data_yaml.exists():
            raise TrainingError(f"Dataset config not found: {data_yaml}")

        project_dir.mkdir(parents=True, exist_ok=True)
        try:
            model = YOLO(model_path)
            result = model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=image_size,
                batch=batch_size,
                device=self.device,
                project=str(project_dir),
                name=run_name,
                pretrained=True,
                verbose=True,
            )
        except Exception as exc:
            raise TrainingError(f"YOLO training failed: {exc}") from exc

        save_dir = Path(getattr(result, "save_dir", project_dir / run_name))
        best_weights = save_dir / "weights" / "best.pt"
        if best_weights.exists():
            return best_weights

        last_weights = save_dir / "weights" / "last.pt"
        if last_weights.exists():
            return last_weights

        raise TrainingError(f"Training finished but no weights found in {save_dir}.")
