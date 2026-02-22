@echo off
setlocal
python vision_ai\main.py run --camera 0 --width 1280 --height 720 --fps 24 --queue-size 3 --yolo-model yolov8s.pt --accuracy-mode max %*
