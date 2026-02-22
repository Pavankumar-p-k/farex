# Farex: AI Facial + Object Recognition Platform

## New: Distributed Hybrid Platform
For Windows + Android + Local FastAPI + Supabase cloud architecture, see:
`distributed_platform/README.md`

## New: Integrated Vision AI Platform
For enterprise-grade integrated runtime (face + YOLO + OCR with multithreaded pipeline), use:
```bash
python vision_ai/main.py run --camera 0 --yolo-model yolov8s.pt
```
Full setup and architecture docs:
`vision_ai/README.md`

Farex is a professional Python system for:
- live multi-person face recognition,
- hologram-style live skeleton tracking,
- web-based unknown-user enrollment and permanent saving to SQLite,
- real-time webcam OCR for printed and digital text,
- real-time YOLOv8 object detection with custom training workflow.

## Features
- Register people directly from webcam (no pre-saved images).
- Auto-capture multiple face samples.
- Generate embeddings and store averaged face encoding.
- SQLite-backed face identity records.
- Real-time face recognition of different people.
- Professional control-suite dashboard with source manager and mobile access links.
- Automatic camera source detection (USB/mobile webcam drivers) with runtime switching.
- Separate live streams in web UI:
  - camera recognition feed
  - skeleton-only hologram feed
- Futuristic web dashboard with no `cv2.imshow` window.
- Unknown face card in UI with save form (name + ID) for permanent storage.
- GPU-aware inference path (CUDA) for RTX-class systems, with CPU fallback.
- Real-time OCR with text region detection, tilted-text handling, and low-light enhancement.
- Real-time YOLOv8 object detection with multi-object tracking view (boxes + labels + confidence).
- Custom YOLO dataset YAML generator and CLI training command.

## Project Structure
```text
face_attendance_system/
  run.py
  serve_web.py
  requirements.txt
  README.md
  data/
  logs/
  face_attendance/
    __init__.py
    attendance_service.py
    camera.py
    config.py
    database.py
    exceptions.py
    face_engine.py
    logger.py
    object_detector.py
    object_detection_runtime.py
    object_training.py
    object_types.py
    ocr_engine.py
    ocr_preprocessor.py
    ocr_persistence.py
    ocr_runtime.py
    ocr_types.py
    recognition_only_service.py
    recognition_service.py
    registration_service.py
    web_app.py
    web_runtime.py
  web/
    templates/
      index.html
    static/
      styles.css
      app.js
```

## Setup
```bash
cd face_attendance_system
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### One Command (Run All Dashboards)
From project root:
```bash
app.cmd
```
High-FPS preset:
```bash
turbo.cmd
```
or:
```bash
powershell -ExecutionPolicy Bypass -File .\app.ps1
```

What it starts:
- main dashboard server (`run.py web`) on `http://localhost:8000`
- mobile preview server (`mobile_dashboard/www`) on `http://localhost:8100`

`turbo.cmd` uses a 640x480@120 profile and disables skeleton/object/OCR in the web dashboard path for maximum FPS.

Optional flags:
```bash
app.cmd --camera 0 --dashboard-port 8000 --mobile-port 8100
```

Persistent settings:
- `app.cmd` now loads `.env` from project root automatically.
- Edit `.env` to keep your preferred camera auto-switch/profile options.

Google Lens style mode (face + objects + text in one live view):
```bash
lens.cmd
```
High-accuracy Lens mode (stronger OCR + object recall, slower):
```bash
lens_pro.cmd
```

### 1) Register Employee
```bash
python run.py register --id EMP001 --name "Alice Johnson" --samples 24 --camera 0
```

### 2) Start Web Dashboard (Recommended)
```bash
python run.py web --host 0.0.0.0 --port 8000
```
Open:
`http://localhost:8000`

Optional camera override:
```bash
python run.py web --host 0.0.0.0 --port 8000 --camera 1
```

Camera auto-selection environment flags:
```bash
set FACE_CAMERA_AUTOSELECT=1
set FACE_CAMERA_PREFER_HIGHEST_INDEX=1
set FACE_CAMERA_SCAN_MAX_INDEX=8
set FACE_CAMERA_HOTPLUG_SCAN_SECONDS=8
```

### Use Mobile As Laptop Camera (Windows)
1. Install a phone webcam app:
   - DroidCam (Android/iOS)
   - Iriun Webcam (Android/iOS)
2. Install the companion desktop driver/client on your laptop.
3. Connect phone + laptop to same Wi-Fi (or USB for lower latency), then start the phone camera stream.
4. Find the camera index exposed by the driver (optional - dashboard can auto-detect):
```bash
python - <<'PY'
import cv2
for i in range(8):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok, _ = cap.read()
    cap.release()
    if ok:
        print("camera index:", i)
PY
```
5. Start the dashboard using that camera index:
```bash
set FACE_CAMERA_INDEX=1
python -m uvicorn face_attendance.web_app:app --host 0.0.0.0 --port 8000
```
You can also tune runtime quality:
```bash
set FACE_FRAME_WIDTH=848
set FACE_FRAME_HEIGHT=480
set FACE_FRAME_FPS=20
set FACE_TARGET_LOOP_FPS=20
set FACE_JPEG_QUALITY=78
set FACE_FACE_INFERENCE_STRIDE=2
set FACE_POSE_INFERENCE_STRIDE=2
# Optional (higher CPU load):
set FACE_ENABLE_OBJECT_DETECTION=1
set FACE_OBJECT_DETECTION_INTERVAL=5
set FACE_ENABLE_OCR=1
set FACE_OCR_INTERVAL=8
```

### 3) Start CLI Recognition (Optional)
```bash
python run.py recognize --camera 0 --threshold 0.70
```

### 4) Start Real-Time OCR
```bash
python run.py ocr --camera 0 --languages en --output logs/ocr_detected_text.txt
```
Controls:
- `S` saves current detected text snapshot to file.
- `Q` exits OCR mode.

Useful tuning flags:
```bash
python run.py ocr --camera 0 --width 1280 --height 720 --fps 30 --interval 2 --min-confidence 0.35
```

OCR benchmark mode:
```bash
python run.py ocr --camera 0 --benchmark-seconds 30
```

### 5) Start Real-Time YOLOv8 Object Detection
```bash
python run.py detect-objects --camera 0 --model yolov8n.pt --conf 0.25 --iou 0.45
```
Features in live view:
- multiple object detection per frame
- bounding boxes + labels + confidence
- FPS counter + per-frame inference time
- GPU device/model status overlay
- modular service API for face pipeline integration (`ObjectDetectionService.process_frame`)

RTX 4050 benchmark example:
```bash
python run.py detect-objects --camera 0 --model yolov8s.pt --imgsz 640 --benchmark-seconds 30
```

### 6) Custom Object Training (YOLOv8)
Create a dataset YAML:
```bash
python run.py init-object-data --output data/object_data.yaml --train data/images/train --val data/images/val --names person helmet vest
```

Train a custom model:
```bash
python run.py train-objects --data data/object_data.yaml --model yolov8n.pt --epochs 80 --imgsz 640 --batch 16
```

### 7) List Employees
```bash
python run.py list-employees
```

## Web Enrollment Flow
1. Launch dashboard.
2. Unknown person appears in camera stream.
3. "Unknown Face Enrollment" card shows preview + sample count.
4. Fill `Employee ID` and `Full Name`.
5. Click **Save New User**.
6. User is stored permanently in SQLite and available for recognition instantly.

## SQLite Schema Notes
- `employees`: `employee_id`, `name`, averaged `face_encoding`, metadata.

## RTX 4050 Optimization Notes
- Uses CUDA automatically if available (`torch.cuda.is_available()`).
- Uses cuDNN benchmark mode for stable webcam input shapes.
- Uses mixed precision (`torch.cuda.amp.autocast`) in embedding forward pass.
- OCR mode uses EasyOCR with CUDA when available and frame-interval throttling (`--interval`) for higher live FPS.
- YOLOv8 uses CUDA (`cuda:0`) automatically in live detection and training unless `--no-gpu` is used.

## Operational Tips
- Ensure good frontal lighting during registration.
- Keep one face in frame during registration.
- Re-register users if appearance changes significantly.

## Build Installers

### Windows EXE
```bash
powershell -ExecutionPolicy Bypass -File scripts/build_windows_exe.ps1
```
Output:
`dist/FaceAttendanceDashboard.exe`

### Android APK (Dashboard Companion App)
```bash
powershell -ExecutionPolicy Bypass -File scripts/build_android_apk.ps1
```
Output:
`mobile_dashboard/android/app/build/outputs/apk/debug/app-debug.apk`
