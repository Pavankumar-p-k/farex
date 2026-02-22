from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

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
        raise ValueError("class_names cannot be empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"train: {train_path}", f"val: {val_path}"]
    if test_path:
        lines.append(f"test: {test_path}")
    lines.append("")
    lines.append("names:")
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


class YOLOTrainer:
    def __init__(self, prefer_gpu: bool = True) -> None:
        if YOLO is None:
            raise RuntimeError("ultralytics is required. Install with: pip install ultralytics")
        self.device = "cuda:0" if (prefer_gpu and torch.cuda.is_available()) else "cpu"

    def train(
        self,
        data_yaml: Path,
        model_path: str = "yolov8n.pt",
        epochs: int = 100,
        image_size: int = 640,
        batch_size: int = 16,
        project: Path = Path("vision_ai/models/training_runs"),
        run_name: str = "custom",
    ) -> Path:
        if not data_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

        project.mkdir(parents=True, exist_ok=True)
        model = YOLO(model_path)
        train_result = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            device=self.device,
            project=str(project),
            name=run_name,
            pretrained=True,
            verbose=True,
        )
        save_dir = Path(getattr(train_result, "save_dir", project / run_name))
        best = save_dir / "weights" / "best.pt"
        if best.exists():
            return best
        last = save_dir / "weights" / "last.pt"
        if last.exists():
            return last
        raise RuntimeError(f"Training completed but weights not found in {save_dir}")
