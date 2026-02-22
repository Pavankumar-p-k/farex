from __future__ import annotations

import os


def configure_runtime_environment() -> None:
    # Avoid noisy Qt logs in terminal when OpenCV or Qt-enabled dependencies initialize.
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
    # Prefer native window backend on Windows to reduce Qt painter conflicts.
    os.environ.setdefault("OPENCV_UI_BACKEND", "WIN32")
