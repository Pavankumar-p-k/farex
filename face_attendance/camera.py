import cv2

from .camera_capture import open_camera_capture
from .config import FRAME_FPS, FRAME_HEIGHT, FRAME_WIDTH
from .exceptions import CameraError


class CameraStream:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.backend_name: str | None = None

    def __enter__(self) -> "CameraStream":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        self.cap, self.backend_name = open_camera_capture(self.camera_index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)

        cv2.setUseOptimized(True)

    def read(self):
        if self.cap is None:
            raise CameraError("Webcam stream is not initialized.")

        success, frame = self.cap.read()
        if not success or frame is None:
            raise CameraError("Failed to read frame from webcam.")
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
