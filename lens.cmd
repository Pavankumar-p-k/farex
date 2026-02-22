@echo off
setlocal
python vision_ai\main.py run --camera 0 --width 960 --height 540 --fps 20 --queue-size 2 --yolo-model yolov8s.pt --accuracy-mode balanced %*
