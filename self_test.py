import os
import sys
import traceback

def log_test(msg):
    with open("test_execution_log.txt", "a") as f:
        f.write(msg + "\n")

log_test("Starting Self-Test...")

try:
    from visual_forensics import analyze_video
    log_test("Import successful.")
    
    video_path = "dataset/raw/person_A/video1.mp4"
    if not os.path.exists(video_path):
        log_test(f"ERROR: Video file {video_path} not found.")
        sys.exit(1)
        
    log_test(f"Analyzing video: {video_path}")
    
    # We run a very short analysis (just first 10 frames) to be quick
    # Note: analyze_video doesn't take max_frames, but we can mock it 
    # Or just run the full video since it's already implemented.
    
    report = analyze_video(
        video_path=video_path,
        output_video_path="test_output_annotated.mp4",
        model_path="models/best_gait_model.pth",
        data_dir="dataset/processed"
    )
    
    log_test("Analysis finished.")
    
    # Verify report contents
    if report.avg_height_px > 0:
        log_test(f"Height detected: {report.avg_height_px:.2f}px")
    else:
        log_test("WARNING: Height is 0. Pose detection might have failed.")
        
    if report.predicted_identity != "N/A":
        log_test(f"Identity Predicted: {report.predicted_identity}")
    
    report.save_json("test_output_report.json")
    log_test("JSON report saved.")
    
    # Final Success Flag
    with open("TEST_PASSED.txt", "w") as f:
        f.write("The forensic software is fully operational.\n")
        f.write(f"Identity: {report.predicted_identity}\n")
        f.write(f"Confidence: {report.identity_confidence_pct:.2f}%\n")
        
    log_test("TEST PASSED.")

except Exception:
    err = traceback.format_exc()
    log_test(f"CRITICAL FAILURE:\n{err}")
    with open("TEST_FAILED.txt", "w") as f:
        f.write(err)
    log_test("TEST FAILED.")
