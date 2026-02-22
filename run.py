import argparse
import sys
from pathlib import Path

import uvicorn

from face_attendance.config import DB_PATH, DEVICE, RECOGNITION_THRESHOLD, REGISTRATION_SAMPLES
from face_attendance.database import AttendanceDatabase
from face_attendance.exceptions import AttendanceError
from face_attendance.face_engine import FaceEngine
from face_attendance.logger import setup_logger
from face_attendance.object_detection_runtime import ObjectDetectionConfig, ObjectDetectionRuntime
from face_attendance.object_training import YOLOTrainingService, write_dataset_yaml
from face_attendance.ocr_runtime import OCRRuntimeConfig, RealTimeOCRSystem
from face_attendance.recognition_only_service import FaceRecognitionService
from face_attendance.registration_service import RegistrationService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live Face Registration and Recognition System"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register", help="Register or update an employee face profile")
    register.add_argument("--id", required=True, dest="employee_id", help="Employee ID")
    register.add_argument("--name", required=True, help="Employee name")
    register.add_argument("--samples", type=int, default=REGISTRATION_SAMPLES, help="Number of face samples")
    register.add_argument("--camera", type=int, default=0, help="Webcam index")

    recognize = subparsers.add_parser("recognize", help="Run real-time face recognition of different persons")
    recognize.add_argument("--camera", type=int, default=0, help="Webcam index")
    recognize.add_argument(
        "--threshold",
        type=float,
        default=RECOGNITION_THRESHOLD,
        help="Cosine similarity threshold for recognition",
    )

    web = subparsers.add_parser("web", help="Launch web hologram dashboard (no cv2 window)")
    web.add_argument("--host", default="0.0.0.0", help="Host interface")
    web.add_argument("--port", type=int, default=8000, help="Port")
    web.add_argument("--camera", type=int, default=None, help="Camera index override")

    ocr = subparsers.add_parser("ocr", help="Run real-time webcam OCR with live overlay")
    ocr.add_argument("--camera", type=int, default=0, help="Webcam index")
    ocr.add_argument("--width", type=int, default=1280, help="Capture width")
    ocr.add_argument("--height", type=int, default=720, help="Capture height")
    ocr.add_argument("--fps", type=int, default=30, help="Capture FPS")
    ocr.add_argument(
        "--interval",
        type=int,
        default=2,
        help="Run OCR every N frames for speed (1 = every frame)",
    )
    ocr.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="EasyOCR language codes (example: en, en hi)",
    )
    ocr.add_argument(
        "--output",
        type=Path,
        default=Path("logs/ocr_detected_text.txt"),
        help="File path for saved text snapshots",
    )
    ocr.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU inference even if CUDA is available",
    )
    ocr.add_argument(
        "--min-confidence",
        type=float,
        default=0.35,
        help="Minimum confidence threshold for text display/save",
    )
    ocr.add_argument(
        "--benchmark-seconds",
        type=int,
        default=0,
        help="If > 0, run benchmark for N seconds and print summary",
    )

    detect = subparsers.add_parser(
        "detect-objects",
        help="Run real-time YOLOv8 object detection on webcam",
    )
    detect.add_argument("--camera", type=int, default=0, help="Webcam index")
    detect.add_argument("--width", type=int, default=1280, help="Capture width")
    detect.add_argument("--height", type=int, default=720, help="Capture height")
    detect.add_argument("--fps", type=int, default=30, help="Capture FPS")
    detect.add_argument("--model", default="yolov8n.pt", help="YOLO model weights path or name")
    detect.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    detect.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    detect.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    detect.add_argument("--max-det", type=int, default=300, help="Maximum detections per frame")
    detect.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Optional class IDs to filter (example: --classes 0 2 7)",
    )
    detect.add_argument("--no-gpu", action="store_true", help="Force CPU inference")
    detect.add_argument(
        "--benchmark-seconds",
        type=int,
        default=0,
        help="If > 0, run benchmark for N seconds and print summary",
    )

    init_data = subparsers.add_parser(
        "init-object-data",
        help="Generate a YOLO dataset YAML for custom training",
    )
    init_data.add_argument("--output", type=Path, required=True, help="Output YAML path")
    init_data.add_argument("--train", required=True, help="Training images path")
    init_data.add_argument("--val", required=True, help="Validation images path")
    init_data.add_argument("--test", default=None, help="Optional test images path")
    init_data.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Class names in order (example: person helmet vest)",
    )

    train = subparsers.add_parser(
        "train-objects",
        help="Train custom YOLOv8 model on your dataset",
    )
    train.add_argument("--data", type=Path, required=True, help="Path to YOLO dataset YAML")
    train.add_argument("--model", default="yolov8n.pt", help="Base model weights")
    train.add_argument("--epochs", type=int, default=50, help="Training epochs")
    train.add_argument("--imgsz", type=int, default=640, help="Training image size")
    train.add_argument("--batch", type=int, default=16, help="Batch size")
    train.add_argument(
        "--project",
        type=Path,
        default=Path("runs/object_training"),
        help="Directory for training runs",
    )
    train.add_argument("--name", default="custom", help="Run name")
    train.add_argument("--no-gpu", action="store_true", help="Force CPU training")

    list_cmd = subparsers.add_parser("list-employees", help="List registered employees")
    list_cmd.add_argument("--limit", type=int, default=100, help="Max rows to print")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logger = setup_logger("main")

    try:
        if args.command == "register":
            db = AttendanceDatabase(DB_PATH)
            engine = FaceEngine(device=DEVICE)
            service = RegistrationService(db=db, engine=engine)
            service.register_employee(
                employee_id=args.employee_id,
                name=args.name,
                camera_index=args.camera,
                target_samples=args.samples,
            )
            print(f"Registration successful for {args.employee_id} ({args.name}).")
            return 0

        if args.command == "recognize":
            db = AttendanceDatabase(DB_PATH)
            engine = FaceEngine(device=DEVICE)
            service = FaceRecognitionService(db=db, engine=engine, threshold=args.threshold)
            service.run(camera_index=args.camera)
            print("Recognition stopped.")
            return 0

        if args.command == "web":
            from face_attendance.web_app import create_web_app

            app = create_web_app(camera_index=args.camera)
            uvicorn.run(app, host=args.host, port=args.port, log_level="info")
            return 0

        if args.command == "ocr":
            config = OCRRuntimeConfig(
                camera_index=args.camera,
                frame_width=args.width,
                frame_height=args.height,
                frame_fps=args.fps,
                detection_interval=args.interval,
                languages=args.languages,
                prefer_gpu=not args.no_gpu,
                min_confidence=args.min_confidence,
                output_file=args.output,
                benchmark_seconds=args.benchmark_seconds,
            )
            runtime = RealTimeOCRSystem(config=config)
            summary = runtime.run()
            if summary is not None:
                print(
                    "OCR benchmark: "
                    f"duration={summary['duration_seconds']:.2f}s, "
                    f"avg_fps={summary['avg_fps']:.2f}, "
                    f"avg_detection_ms={summary['avg_detection_ms']:.2f}, "
                    f"frames={int(summary['frames'])}"
                )
            print("OCR session stopped.")
            return 0

        if args.command == "detect-objects":
            config = ObjectDetectionConfig(
                camera_index=args.camera,
                frame_width=args.width,
                frame_height=args.height,
                frame_fps=args.fps,
                model_path=args.model,
                prefer_gpu=not args.no_gpu,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                classes=args.classes,
                image_size=args.imgsz,
                max_detections=args.max_det,
                benchmark_seconds=args.benchmark_seconds,
            )
            runtime = ObjectDetectionRuntime(config=config)
            summary = runtime.run()
            if summary is not None:
                print(
                    "Object benchmark: "
                    f"duration={summary['duration_seconds']:.2f}s, "
                    f"avg_fps={summary['avg_fps']:.2f}, "
                    f"avg_inference_ms={summary['avg_inference_ms']:.2f}, "
                    f"frames={int(summary['frames'])}"
                )
            print("Object detection session stopped.")
            return 0

        if args.command == "init-object-data":
            output_path = write_dataset_yaml(
                output_path=args.output,
                train_path=args.train,
                val_path=args.val,
                class_names=args.names,
                test_path=args.test,
            )
            print(f"Created YOLO dataset config: {output_path}")
            return 0

        if args.command == "train-objects":
            trainer = YOLOTrainingService(prefer_gpu=not args.no_gpu)
            best_weights = trainer.train(
                data_yaml=args.data,
                model_path=args.model,
                epochs=args.epochs,
                image_size=args.imgsz,
                batch_size=args.batch,
                project_dir=args.project,
                run_name=args.name,
            )
            print(f"Training complete. Best weights: {best_weights}")
            return 0

        if args.command == "list-employees":
            db = AttendanceDatabase(DB_PATH)
            records = db.list_employees()
            if not records:
                print("No employees registered.")
                return 0

            print(f"{'Employee ID':<16} {'Name'}")
            print("-" * 52)
            for record in records[: args.limit]:
                print(f"{record.employee_id:<16} {record.name}")
            return 0

    except AttendanceError as exc:
        logger.error("Application error: %s", exc)
        print(f"Error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 1
    except Exception as exc:
        logger.exception("Unexpected failure")
        print(f"Unexpected error: {exc}")
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
