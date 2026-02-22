# Vision AI Platform (RTX 4050 Optimized)

Production-grade real-time platform integrating:
- Face recognition with live ArcFace registration (InsightFace)
- YOLOv8 object detection
- Real-time OCR (EasyOCR)
- Multi-threaded non-blocking pipeline
- Structured logging, attendance persistence, and CSV export

## Folder Design
```text
vision_ai/
├── main.py
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── .embedding.key         # auto-generated if env key not provided
├── face_module/
│   ├── __init__.py
│   ├── arcface_engine.py
│   └── recognition_service.py
├── object_module/
│   ├── __init__.py
│   ├── yolo_detector.py
│   └── trainer.py
├── ocr_module/
│   ├── __init__.py
│   └── ocr_engine.py
├── database/
│   ├── __init__.py
│   └── storage.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── metrics.py
│   ├── queueing.py
│   ├── runtime.py
│   └── types.py
├── models/
└── logs/
```

## Architecture
- `camera-thread`: captures frames and pushes latest frame to module queues.
- `face-thread`: ArcFace embeddings + matching + attendance marking.
- `object-thread`: YOLOv8 inference.
- `ocr-thread`: text detection/recognition.
- `aggregation-thread`: merges module outputs + handles runtime commands.
- Main thread: rendering only (`cv2.imshow`) for stable GUI behavior.

Queues are bounded and always keep the latest frame (`put_latest`) to avoid blocking and stale backlogs.

## CUDA Setup (Windows / RTX 4050)
1. Install NVIDIA driver and verify:
```powershell
nvidia-smi
```
2. Create venv and install dependencies:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r vision_ai/requirements.txt
```
`insightface` is optional on Windows/Python 3.12+ (wheel may be unavailable).  
If installed, ArcFace backend is used automatically; otherwise runtime falls back to OpenCV-based face matching.
3. Validate GPU in Python:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Run Integrated Platform
```powershell
python vision_ai/main.py run --camera 0 --yolo-model yolov8s.pt
```
High-accuracy preset (multi-pass OCR + YOLO ensemble + temporal stabilization):
```powershell
python vision_ai/main.py run --camera 0 --yolo-model yolov8s.pt --accuracy-mode max
```

Runtime controls:
- `Q`: quit
- `S`: save current OCR snapshot
- Terminal commands during run:
  - `register EMP001 Alice Johnson`
  - `summary`
  - `export vision_ai/logs/attendance_today.csv`
  - `quit`

## Object Training Support
Generate dataset YAML:
```powershell
python vision_ai/main.py init-object-data --output vision_ai/models/object_data.yaml --train data/images/train --val data/images/val --names person helmet vest
```

Train YOLOv8:
```powershell
python vision_ai/main.py train-objects --data vision_ai/models/object_data.yaml --model yolov8n.pt --epochs 80 --imgsz 640 --batch 16
```

## Attendance Export
```powershell
python vision_ai/main.py export-attendance --output vision_ai/logs/attendance_export.csv
```

## Enterprise Notes
- Embeddings are encrypted at rest using Fernet (`cryptography`).
- Attendance uses daily uniqueness (`employee_id + attendance_date`) to prevent duplicates.
- RBAC schema is initialized (`roles`, `users`) for future auth integration.
- Logs are JSON structured (`vision_ai/logs/vision_ai.log`) for SIEM ingestion.
- Face recognition uses vectorized cosine similarity for 500+ employee scale.
