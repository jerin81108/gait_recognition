import sys
import os

def log(msg):
    with open("debug_log.txt", "a") as f:
        f.write(msg + "\n")
    print(msg)

log("Starting Debugging Script...")

try:
    import torch
    log(f"Torch Version: {torch.__version__}")
except Exception as e:
    log(f"Torch Error: {e}")

try:
    import cv2
    log(f"OpenCV Version: {cv2.__version__}")
except Exception as e:
    log(f"OpenCV Error: {e}")

try:
    from ultralytics import YOLO
    log("Ultralytics imported.")
except Exception as e:
    log(f"Ultralytics Error: {e}")

try:
    from model import GaitLSTM
    log("model.py loaded.")
    m = GaitLSTM(input_size=69, num_classes=4)
    log(f"Model instantiated with {sum(p.numel() for p in m.parameters()):,} params.")
except Exception as e:
    log(f"model.py Error: {e}")

try:
    from visual_forensics import analyze_video
    log("visual_forensics.py loaded.")
except Exception as e:
    log(f"visual_forensics.py Error: {e}")

log("Debugging Script Finished.")
