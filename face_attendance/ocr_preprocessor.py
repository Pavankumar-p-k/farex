from __future__ import annotations

import cv2
import numpy as np


class FramePreprocessor:
    def __init__(
        self,
        low_light_threshold: float = 95.0,
        clahe_clip_limit: float = 2.5,
    ) -> None:
        self.low_light_threshold = low_light_threshold
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        enhanced = self._apply_low_light_enhancement(frame)
        return self._sharpen(enhanced)

    def _apply_low_light_enhancement(self, frame: np.ndarray) -> np.ndarray:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_luminance = float(np.mean(grayscale))

        boosted = frame
        if mean_luminance < self.low_light_threshold:
            gamma = max(0.45, min(0.9, mean_luminance / self.low_light_threshold))
            boosted = self._apply_gamma(frame, gamma)

        lab = cv2.cvtColor(boosted, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = self.clahe.apply(l_channel)
        merged = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _apply_gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
        lut = np.array(
            [((value / 255.0) ** gamma) * 255.0 for value in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(frame, lut)

    @staticmethod
    def _sharpen(frame: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
        return cv2.addWeighted(frame, 1.2, blurred, -0.2, 0)
