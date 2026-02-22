import atexit
import time
from io import BytesIO
from pathlib import Path
from typing import Iterator

from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user

from .admin_service import AdminService
from .config import BASE_DIR, CAMERA_INDEX, DB_PATH, DEVICE, RECOGNITION_THRESHOLD
from .database import AttendanceDatabase
from .exceptions import AttendanceError
from .face_engine import FaceEngine
from .web_runtime import WebVisionRuntime


WEB_DIR = Path(BASE_DIR) / "web"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


class AdminUser(UserMixin):
    def __init__(self, username: str):
        self.id = username


def _mjpeg_frame_generator(runtime: WebVisionRuntime, stream_name: str) -> Iterator[bytes]:
    while True:
        frame = runtime.get_jpeg_frame(stream_name=stream_name)
        if frame is None:
            time.sleep(0.03)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


def create_flask_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(TEMPLATES_DIR),
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
    )
    app.config["SECRET_KEY"] = "face-attendance-phase2-secret"

    db = AttendanceDatabase(DB_PATH)
    admin_service = AdminService(db)
    admin_service.ensure_default_admin()

    engine = FaceEngine(device=DEVICE)
    runtime = WebVisionRuntime(
        db=db,
        engine=engine,
        threshold=RECOGNITION_THRESHOLD,
        camera_index=CAMERA_INDEX,
    )
    runtime.start()
    atexit.register(runtime.stop)

    login_manager = LoginManager()
    login_manager.login_view = "login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id: str):
        found = db.get_admin_user(user_id)
        return AdminUser(user_id) if found else None

    @app.get("/login")
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        return render_template("login.html")

    @app.post("/login")
    def login_post():
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        if admin_service.verify_login(username, password):
            login_user(AdminUser(username))
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "error")
        return redirect(url_for("login"))

    @app.get("/logout")
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("login"))

    @app.get("/")
    @login_required
    def dashboard():
        return render_template("dashboard.html", stats=admin_service.stats(), user=current_user)

    @app.get("/employees")
    @login_required
    def employees():
        profiles = db.list_employee_profiles()
        return render_template("employees.html", employees=profiles, user=current_user)

    @app.post("/employees/delete/<employee_id>")
    @login_required
    def delete_employee(employee_id: str):
        try:
            admin_service.delete_employee(employee_id)
            runtime.refresh_known_faces()
            flash(f"Employee {employee_id} deleted.", "ok")
        except AttendanceError as exc:
            flash(str(exc), "error")
        return redirect(url_for("employees"))

    @app.get("/attendance")
    @login_required
    def attendance():
        query_text = request.args.get("q", "").strip()
        date_from = request.args.get("from", "").strip()
        date_to = request.args.get("to", "").strip()
        rows = admin_service.attendance_rows(query_text=query_text, date_from=date_from, date_to=date_to, limit=3000)
        return render_template(
            "attendance.html",
            rows=rows,
            q=query_text,
            date_from=date_from,
            date_to=date_to,
            user=current_user,
        )

    @app.get("/attendance/export")
    @login_required
    def attendance_export():
        query_text = request.args.get("q", "").strip()
        date_from = request.args.get("from", "").strip()
        date_to = request.args.get("to", "").strip()
        payload = admin_service.attendance_excel(query_text=query_text, date_from=date_from, date_to=date_to)
        return send_file(
            BytesIO(payload),
            as_attachment=True,
            download_name="attendance_report.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    @app.get("/api/state")
    @login_required
    def state():
        return jsonify(runtime.get_dashboard_state())

    @app.get("/api/stream/<stream_name>")
    @login_required
    def stream(stream_name: str):
        if stream_name not in {"camera", "skeleton"}:
            return jsonify({"error": "Invalid stream."}), 400
        return Response(
            _mjpeg_frame_generator(runtime, stream_name=stream_name),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/api/register-pending")
    @login_required
    def register_pending():
        body = request.get_json(silent=True) or {}
        employee_id = str(body.get("employee_id", ""))
        name = str(body.get("name", ""))
        try:
            saved = runtime.register_pending_unknown(employee_id=employee_id, name=name)
            return jsonify({"ok": True, "saved": saved})
        except AttendanceError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/clear-pending")
    @login_required
    def clear_pending():
        runtime.clear_pending_unknown()
        return jsonify({"ok": True})

    return app
