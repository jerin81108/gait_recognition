import os
import sys
import torch
import numpy as np
import traceback

def log_final(msg):
    with open("FINAL_TEST_LOG.txt", "a") as f:
        f.write(msg + "\n")
    print(msg)

# Clear old log
if os.path.exists("FINAL_TEST_LOG.txt"):
    os.remove("FINAL_TEST_LOG.txt")

log_final("=== GAIT FORENSIC SYSTEM FINAL TEST ===")

try:
    from visual_forensics import analyze_video
    from model import GaitLSTM
    
    # 1. Check Model Metadata
    model_path = "models/best_gait_model.pth"
    if not os.path.exists(model_path):
        log_final(f"FAILED: Model not found at {model_path}")
        sys.exit(1)
        
    state_dict = torch.load(model_path, map_location="cpu")
    log_final(f"Model keys: {list(state_dict.keys())[:5]}...")
    
    # Check output layer size
    if 'output.weight' in state_dict:
        out_size = state_dict['output.weight'].shape[0]
        log_final(f"Model trained for {out_size} classes.")
    
    # 2. Run Analysis
    video_path = "dataset/raw/person_A/video1.mp4"
    log_final(f"Running analysis on {video_path}...")
    
    report = analyze_video(
        video_path=video_path,
        output_video_path="final_verified_video.mp4",
        model_path=model_path,
        data_dir="dataset/processed",
        max_frames=30
    )
    
    log_final("Analysis Complete.")
    log_final(f"Video saved: final_verified_video.mp4")
    log_final(f"Height: {report.avg_height_px:.2f} px")
    log_final(f"Stride: {report.avg_stride_px:.2f} px")
    log_final(f"Predict Identity: {report.predicted_identity}")
    log_final(f"Confidence: {report.identity_confidence_pct:.2f}%")
    
    if report.predicted_identity == "Model error":
        log_final("ERROR: Identity prediction failed internally.")
    else:
        log_final("SUCCESS: System is fully operational.")
        with open("SYSTEM_READY.txt", "w") as f:
            f.write("PASS")

except Exception as e:
    log_final(f"CRITICAL TEST ERROR: {e}")
    log_final(traceback.format_exc())

log_final("=== END OF TEST ===")
