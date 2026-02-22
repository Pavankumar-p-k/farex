from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
from contextlib import closing
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import uvicorn

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _local_ipv4_addresses() -> list[str]:
    addresses = {"127.0.0.1"}
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip:
                addresses.add(ip)
    except OSError:
        pass
    return sorted(addresses, key=lambda ip: (ip.startswith("127."), ip))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start all project dashboards with one command.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface for dashboard servers.")
    parser.add_argument("--dashboard-port", type=int, default=8000, help="Main dashboard port.")
    parser.add_argument("--mobile-port", type=int, default=8100, help="Mobile web preview port.")
    parser.add_argument("--camera", type=int, default=None, help="Camera index override for main dashboard.")
    parser.add_argument("--no-mobile", action="store_true", help="Do not start mobile web preview server.")
    return parser


def _port_in_use(host: str, port: int) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _http_reachable(url: str) -> bool:
    try:
        with urlopen(url, timeout=1.5) as response:
            return 200 <= int(response.status) < 500
    except (OSError, URLError, ValueError):
        return False


def _print_links(host: str, dashboard_port: int, mobile_port: int, mobile_enabled: bool) -> None:
    print("\nProject links")
    print("-" * 60)
    ips = _local_ipv4_addresses()
    preferred_host = "127.0.0.1" if host == "0.0.0.0" else host

    print(f"Main Dashboard : http://{preferred_host}:{dashboard_port}")
    print(f"API State      : http://{preferred_host}:{dashboard_port}/api/state")
    print(f"Cameras API    : http://{preferred_host}:{dashboard_port}/api/cameras")
    print(f"Mobile URLs API: http://{preferred_host}:{dashboard_port}/api/mobile-access")
    if mobile_enabled:
        print(f"Mobile Preview : http://{preferred_host}:{mobile_port}")

    lan_ips = [ip for ip in ips if not ip.startswith("127.")]
    if lan_ips:
        print("\nLAN access")
        for ip in lan_ips:
            print(f"- Dashboard: http://{ip}:{dashboard_port}")
            if mobile_enabled:
                print(f"- Mobile   : http://{ip}:{mobile_port}")
    print("-" * 60)


def _start_mobile_server(host: str, mobile_port: int) -> tuple[ThreadingHTTPServer, threading.Thread]:
    mobile_dir = ROOT_DIR / "mobile_dashboard" / "www"
    handler = partial(SimpleHTTPRequestHandler, directory=str(mobile_dir))
    server = ThreadingHTTPServer((host, mobile_port), handler)
    thread = threading.Thread(target=server.serve_forever, name="mobile-web-server", daemon=True)
    thread.start()
    return server, thread


def _apply_stable_camera_defaults() -> None:
    # Keep stream stable and responsive on typical laptop/mobile camera drivers.
    os.environ.setdefault("FACE_FRAME_WIDTH", "848")
    os.environ.setdefault("FACE_FRAME_HEIGHT", "480")
    os.environ.setdefault("FACE_FRAME_FPS", "20")
    os.environ.setdefault("FACE_TARGET_LOOP_FPS", "20")
    os.environ.setdefault("FACE_JPEG_QUALITY", "78")
    os.environ.setdefault("FACE_FACE_INFERENCE_STRIDE", "2")
    os.environ.setdefault("FACE_FACE_INFERENCE_SCALE", "0.85")
    os.environ.setdefault("FACE_POSE_INFERENCE_STRIDE", "2")
    os.environ.setdefault("FACE_POSE_INFERENCE_SCALE", "0.75")
    os.environ.setdefault("FACE_ANALYTICS_MAX_SIDE", "960")
    os.environ.setdefault("FACE_ENABLE_OBJECT_DETECTION", "0")
    os.environ.setdefault("FACE_OBJECT_DETECTION_INTERVAL", "5")
    os.environ.setdefault("FACE_OBJECT_IMAGE_SIZE", "512")
    os.environ.setdefault("FACE_OBJECT_MODEL", "yolov8n.pt")
    os.environ.setdefault("FACE_ENABLE_OCR", "0")
    os.environ.setdefault("FACE_OCR_INTERVAL", "8")


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_wrapping_quotes(value.strip())
        if not key:
            continue

        # Shell/session variables win over .env file values.
        os.environ.setdefault(key, value)


def main() -> int:
    args = _build_parser().parse_args()
    _load_env_file(ROOT_DIR / ".env")
    _apply_stable_camera_defaults()
    probe_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host

    mobile_server: ThreadingHTTPServer | None = None
    try:
        if _port_in_use(probe_host, args.dashboard_port):
            state_url = f"http://{probe_host}:{args.dashboard_port}/api/state"
            if _http_reachable(state_url):
                _print_links(
                    host=args.host,
                    dashboard_port=args.dashboard_port,
                    mobile_port=args.mobile_port,
                    mobile_enabled=not args.no_mobile,
                )
                print(f"[dashboard] Already running at {state_url}")
                return 0
            print(
                f"[dashboard] Port {args.dashboard_port} is already in use on {probe_host}. "
                "Stop the other process or choose a different --dashboard-port."
            )
            return 1

        if not args.no_mobile:
            if _port_in_use(probe_host, args.mobile_port):
                print(
                    f"[mobile] Port {args.mobile_port} is already in use on {probe_host}. "
                    "Skipping mobile preview server for this run."
                )
            else:
                try:
                    mobile_server, _ = _start_mobile_server(args.host, args.mobile_port)
                    print(f"[mobile] Serving mobile preview on {args.host}:{args.mobile_port}")
                except OSError as exc:
                    print(f"[mobile] Failed to start mobile preview server: {exc}")
                    mobile_server = None

        _print_links(
            host=args.host,
            dashboard_port=args.dashboard_port,
            mobile_port=args.mobile_port,
            mobile_enabled=mobile_server is not None,
        )
        print("Press Ctrl+C to stop all services.")

        from face_attendance.web_app import create_web_app

        app = create_web_app(camera_index=args.camera)
        uvicorn.run(app, host=args.host, port=args.dashboard_port, log_level="info")
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if mobile_server is not None:
            mobile_server.shutdown()
            mobile_server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
