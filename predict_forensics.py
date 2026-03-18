"""
predict_forensics.py
====================
Command-line entry point for the visual forensic gait analysis system.

Usage
-----
  python predict_forensics.py <video_path>
      [--model models/best_gait_model.pth]
      [--data_dir dataset/processed]
      [--output_video annotated_output.mp4]
      [--report_json forensic_report.json]

Examples
--------
  # Basic run (console report only)
  python predict_forensics.py suspect_walk.mp4

  # Full run with annotated video + JSON report + custom model
  python predict_forensics.py suspect_walk.mp4 \\
      --output_video annotated_suspect.mp4 \\
      --report_json  report.json \\
      --model        models/best_gait_model.pth \\
      --data_dir     dataset/processed
"""

import os
import argparse
import sys

from visual_forensics import analyze_video


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visual Forensics Prediction System\n"
            "Detects and predicts forensic details from a video using\n"
            "YOLOv8 pose/object detection + LSTM gait identification."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file (.mp4 / .avi / .mov / .mkv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join("models", "best_gait_model.pth"),
        help="Path to the trained LSTM model weights (.pth). "
             "If not found, identity prediction is skipped. "
             "(default: models/best_gait_model.pth)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("dataset", "processed"),
        help="Path to the processed dataset directory "
             "(used for label mapping). "
             "(default: dataset/processed)",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default=None,
        help="If specified, save an annotated output video to this path. "
             "Recommended extension: .mp4",
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default=None,
        help="If specified, save a machine-readable JSON report to this path.",
    )

    args = parser.parse_args()

    # ---- Validate input ----
    if not os.path.exists(args.video_path):
        print(f"[ERROR] Video file not found: '{args.video_path}'")
        sys.exit(1)

    model_path = args.model if os.path.exists(args.model) else None
    if model_path is None:
        print(
            f"[WARNING] Model weights not found at '{args.model}'. "
            "Identity prediction will be skipped. "
            "Run train.py first to enable it."
        )

    # ---- Run visual forensics analysis ----
    print(f"\n[Forensics] Analysing video: {args.video_path}")
    if args.output_video:
        print(f"[Forensics] Annotated video will be saved to: {args.output_video}")

    report = analyze_video(
        video_path=args.video_path,
        output_video_path=args.output_video,
        model_path=model_path,
        data_dir=args.data_dir,
    )

    # ---- Print to console ----
    report.print_report()

    # ---- Save JSON ----
    if args.report_json:
        report.save_json(args.report_json)


if __name__ == "__main__":
    main()
