import argparse

import uvicorn

from face_attendance.web_app import create_web_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face Attendance Dashboard Launcher")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--camera", type=int, default=None, help="Camera index override")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = create_web_app(camera_index=args.camera)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
