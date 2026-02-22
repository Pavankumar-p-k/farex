class AttendanceError(Exception):
    """Base exception for the attendance system."""


class CameraError(AttendanceError):
    """Raised when webcam access fails."""


class FaceEngineError(AttendanceError):
    """Raised when face detection or embedding generation fails."""


class DatabaseError(AttendanceError):
    """Raised when database operations fail."""


class OCRError(AttendanceError):
    """Raised when OCR pipeline initialization or inference fails."""


class DetectorError(AttendanceError):
    """Raised when object detection initialization or inference fails."""


class TrainingError(AttendanceError):
    """Raised when object detector training setup or execution fails."""
