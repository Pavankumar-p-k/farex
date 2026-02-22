from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

from .ocr_types import OCRDetection


class OCRTextSaver:
    def __init__(self, output_path: Path, min_confidence: float = 0.35) -> None:
        self.output_path = output_path
        self.min_confidence = min_confidence
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, detections: Iterable[OCRDetection]) -> int:
        unique_lines: list[str] = []
        seen: set[str] = set()
        for detection in detections:
            if detection.confidence < self.min_confidence:
                continue
            text = detection.text.strip()
            normalized = text.lower()
            if not text or normalized in seen:
                continue
            seen.add(normalized)
            unique_lines.append(text)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}]\n")
            if unique_lines:
                for line in unique_lines:
                    handle.write(f"{line}\n")
            else:
                handle.write("(no high-confidence text)\n")
            handle.write("\n")

        return len(unique_lines)
