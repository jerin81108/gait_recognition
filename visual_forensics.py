"""
visual_forensics.py
===================
Core visual forensics engine for the gait analysis system.

Responsibilities
----------------
* Run YOLOv8 pose + object detection on every frame of a video.
* Derive rich forensic metrics per frame:
    - Skeleton-based: height, stride, shoulder/hip width, arm/leg lengths
    - Clothing dominant color (K-means on torso region)
    - Postural lean angle (torso vs. vertical axis)
    - Head-to-body ratio
    - Step velocity (inter-frame ankle displacement, px/s)
    - Arm-swing asymmetry index
    - Accessory presence (backpack, umbrella, handbag, tie, suitcase)
* Aggregate frame statistics into a ForensicReport dataclass.
* Write an annotated output video (optional).
* Export a structured JSON report (optional).

Usage
-----
    from visual_forensics import analyze_video
    report = analyze_video("input.mp4", output_video_path="annotated.mp4")
    report.print_report()
    report.save_json("report.json")
"""

import os
import json
import math
import time
import cv2
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# YOLOv8 model paths (same as extract_features.py)
# ---------------------------------------------------------------------------
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_POSE_WEIGHTS = os.path.join(_MODEL_DIR, "yolov8n-pose.pt")
_OBJ_WEIGHTS  = os.path.join(_MODEL_DIR, "yolov8n.pt")

# Lazily loaded so import is fast
_pose_model: Optional[YOLO] = None
_obj_model:  Optional[YOLO] = None

def _get_models():
    global _pose_model, _obj_model
    if _pose_model is None:
        _pose_model = YOLO(_POSE_WEIGHTS)
    if _obj_model is None:
        _obj_model  = YOLO(_OBJ_WEIGHTS)
    return _pose_model, _obj_model

# ---------------------------------------------------------------------------
# COCO class IDs for forensically relevant accessories
# ---------------------------------------------------------------------------
ACCESSORY_CLASSES = {
    24: "Backpack",
    25: "Umbrella",
    26: "Handbag",
    27: "Tie",
    28: "Suitcase",
}

# COCO 17-keypoint indices (for readability)
KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6,
    "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10,
    "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14,
    "l_ankle": 15, "r_ankle": 16,
}

CONF_THRESH = 0.3  # keypoint confidence threshold

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _kp_dist(kpts, i, j):
    """Euclidean distance between keypoints i and j if both confident."""
    if kpts[i][2] > CONF_THRESH and kpts[j][2] > CONF_THRESH:
        return float(np.linalg.norm(kpts[i][:2] - kpts[j][:2]))
    return 0.0


def _midpoint(kpts, i, j):
    """Pixel midpoint between two confident keypoints, else None."""
    if kpts[i][2] > CONF_THRESH and kpts[j][2] > CONF_THRESH:
        return ((kpts[i][0] + kpts[j][0]) / 2, (kpts[i][1] + kpts[j][1]) / 2)
    return None


def _get_dominant_color(frame: np.ndarray, bbox) -> Optional[Tuple[int,int,int]]:
    """
    K-means (K=3) on the torso region of a bounding box.
    Returns the dominant colour as (R, G, B) or None.
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h = y2 - y1
    # Focus on the middle 40% of the box height (torso)
    torso_y1 = y1 + int(h * 0.30)
    torso_y2 = y1 + int(h * 0.70)
    torso_x1 = x1 + int((x2 - x1) * 0.15)
    torso_x2 = x2 - int((x2 - x1) * 0.15)

    crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
    if crop.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pixels = crop_rgb.reshape(-1, 3).astype(np.float32)
    if len(pixels) < 3:
        return None

    k = min(3, len(pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    # Most common cluster
    counts = np.bincount(labels.flatten())
    dominant = centers[np.argmax(counts)]
    return (int(dominant[0]), int(dominant[1]), int(dominant[2]))  # R, G, B


def _estimate_posture(kpts) -> float:
    """
    Torso lean angle (degrees from vertical, 0 = perfectly upright).
    Positive = leaning right, Negative = leaning left.
    """
    sh_mid = _midpoint(kpts, KP["l_shoulder"], KP["r_shoulder"])
    hp_mid = _midpoint(kpts, KP["l_hip"], KP["r_hip"])
    if sh_mid is None or hp_mid is None:
        return 0.0
    dx = sh_mid[0] - hp_mid[0]
    dy = hp_mid[1] - sh_mid[1]  # positive downward in image coords
    if dy == 0:
        return 0.0
    angle = math.degrees(math.atan(dx / dy))
    return angle


def _estimate_head_ratio(kpts, height: float) -> float:
    """
    Head height / total body height.
    Typical value 0.12–0.16 for adults.
    """
    if height <= 0:
        return 0.0
    nose = kpts[KP["nose"]]
    l_ear = kpts[KP["l_ear"]]
    r_ear = kpts[KP["r_ear"]]
    valid_ears = [e for e in [l_ear, r_ear] if e[2] > CONF_THRESH]
    if not valid_ears or nose[2] <= CONF_THRESH:
        return 0.0
    avg_ear_y = float(np.mean([e[1] for e in valid_ears]))
    head_h = abs(float(nose[1]) - avg_ear_y) * 2.5  # approximate full head
    return head_h / height


def _ankle_midpoint(kpts):
    """Mean x,y of visible ankles."""
    pts = []
    for ki in [KP["l_ankle"], KP["r_ankle"]]:
        if kpts[ki][2] > CONF_THRESH:
            pts.append(kpts[ki][:2])
    if pts:
        return np.mean(pts, axis=0)
    return None


def _arm_swing_asymmetry(kpts) -> float:
    """
    |left_arm_angle - right_arm_angle| in degrees.
    0 = perfectly symmetric, large = asymmetric.
    """
    def arm_angle(sh_idx, el_idx):
        sh = kpts[sh_idx]
        el = kpts[el_idx]
        if sh[2] > CONF_THRESH and el[2] > CONF_THRESH:
            dx = float(el[0] - sh[0])
            dy = float(el[1] - sh[1])
            return math.degrees(math.atan2(dy, dx))
        return None

    l_ang = arm_angle(KP["l_shoulder"], KP["l_elbow"])
    r_ang = arm_angle(KP["r_shoulder"], KP["r_elbow"])
    if l_ang is not None and r_ang is not None:
        return abs(l_ang - r_ang)
    return 0.0


def _color_name(rgb: Optional[Tuple]) -> str:
    """Map dominant RGB to a human-readable colour name (heuristic)."""
    if rgb is None:
        return "Unknown"
    r, g, b = rgb
    h, s, v = cv2.cvtColor(
        np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV
    )[0][0]
    if v < 50:
        return "Black"
    if v > 200 and s < 40:
        return "White"
    if s < 40:
        return "Gray"
    if h < 10 or h > 160:
        return "Red"
    if 10 <= h < 25:
        return "Orange"
    if 25 <= h < 35:
        return "Yellow"
    if 35 <= h < 85:
        return "Green"
    if 85 <= h < 125:
        return "Blue"
    if 125 <= h < 145:
        return "Purple"
    return "Mixed"


# ---------------------------------------------------------------------------
# ForensicReport dataclass
# ---------------------------------------------------------------------------

@dataclass
class ForensicReport:
    video_path: str = ""
    total_frames: int = 0
    detection_frames: int = 0

    # Physical metrics (pixel-space averages)
    avg_height_px: float = 0.0
    avg_stride_px: float = 0.0
    avg_shoulder_width_px: float = 0.0
    avg_hip_width_px: float = 0.0
    avg_arm_length_px: float = 0.0
    avg_leg_length_px: float = 0.0

    # Derived ratios / predictions
    shoulder_hip_ratio: float = 0.0
    stride_height_ratio: float = 0.0
    leg_height_ratio: float = 0.0
    avg_head_ratio: float = 0.0

    # Behavioral / postural
    avg_posture_angle_deg: float = 0.0
    avg_arm_swing_asymmetry_deg: float = 0.0
    avg_step_velocity_px_per_s: float = 0.0

    # Appearance
    dominant_clothing_rgb: Optional[Tuple] = None
    dominant_clothing_color_name: str = "Unknown"

    # Accessories
    accessories_detected: List[str] = field(default_factory=list)

    # Inferred attributes
    predicted_gender: str = "Unknown"
    predicted_gait_type: str = "Unknown"
    predicted_speed_category: str = "Unknown"
    arm_swing_quality: str = "Unknown"
    body_proportions: str = "Unknown"
    posture_description: str = "Unknown"
    head_size_description: str = "Unknown"
    predicted_weight_category: str = "Unknown"

    # Identity (from LSTM model)
    predicted_identity: str = "N/A"
    identity_confidence_pct: float = 0.0

    # Processing time
    processing_time_s: float = 0.0

    # -----------------------------------------------------------------------

    def _derive_attributes(self):
        """Compute all purely derived / label fields from raw metrics."""
        # Gender (shoulder-hip ratio heuristic)
        # Only use heuristic if model prediction is N/A or Unknown
        if self.predicted_gender in ["Unknown", "N/A"] and self.avg_hip_width_px > 0:
            self.shoulder_hip_ratio = self.avg_shoulder_width_px / self.avg_hip_width_px
            shr = self.shoulder_hip_ratio
            if shr > 1.15: # Tuned threshold for male
                self.predicted_gender = "Male (Heuristic)"
            else:
                self.predicted_gender = "Female (Heuristic)"

        # Gait type (stride-to-height ratio)
        if self.avg_height_px > 0:
            self.stride_height_ratio = self.avg_stride_px / self.avg_height_px
            sthr = self.stride_height_ratio
            if sthr > 0.60:
                self.predicted_gait_type = f"Running / Fast Pace  (STHR={sthr:.2f})"
            elif sthr < 0.28:
                self.predicted_gait_type = f"Shuffling / Slow     (STHR={sthr:.2f})"
            else:
                self.predicted_gait_type = f"Normal Walking       (STHR={sthr:.2f})"

            # Leg-to-height ratio
            self.leg_height_ratio = self.avg_leg_length_px / self.avg_height_px
            lhr = self.leg_height_ratio
            if lhr > 0.52:
                self.body_proportions = f"Long Legs  (LHR={lhr:.2f})"
            elif 0 < lhr < 0.44:
                self.body_proportions = f"Short Legs (LHR={lhr:.2f})"
            else:
                self.body_proportions = f"Average    (LHR={lhr:.2f})"

        # Step speed category
        v = self.avg_step_velocity_px_per_s
        if v < 1:
            self.predicted_speed_category = "Stationary"
        elif v < 40:
            self.predicted_speed_category = f"Slow  ({v:.1f} px/s)"
        elif v < 100:
            self.predicted_speed_category = f"Normal ({v:.1f} px/s)"
        else:
            self.predicted_speed_category = f"Fast  ({v:.1f} px/s)"

        # Arm swing quality
        asym = self.avg_arm_swing_asymmetry_deg
        if asym < 10:
            self.arm_swing_quality = f"Symmetric     (asym={asym:.1f}°)"
        elif asym < 25:
            self.arm_swing_quality = f"Mildly Asymm. (asym={asym:.1f}°)"
        else:
            self.arm_swing_quality = f"Asymmetric    (asym={asym:.1f}°)"

        # Posture
        angle = abs(self.avg_posture_angle_deg)
        side  = "right" if self.avg_posture_angle_deg >= 0 else "left"
        if angle < 5:
            self.posture_description = f"Upright ({angle:.1f}°)"
        elif angle < 15:
            self.posture_description = f"Slight lean {side} ({angle:.1f}°)"
        else:
            self.posture_description = f"Significant lean {side} ({angle:.1f}°)"

        # Head ratio
        hr = self.avg_head_ratio
        if hr == 0:
            self.head_size_description = "Unknown"
        elif hr < 0.12:
            self.head_size_description = f"Small relative head (HHR={hr:.3f})"
        elif hr > 0.18:
            self.head_size_description = f"Large relative head (HHR={hr:.3f})"
        else:
            self.head_size_description = f"Proportionate head  (HHR={hr:.3f})"

        # Clothing colour name
        self.dominant_clothing_color_name = _color_name(self.dominant_clothing_rgb)

        # Weight category (Heuristic based on Volume Proxy: Height * (Shoulder + Hip)/2)
        if self.avg_height_px > 0 and (self.avg_shoulder_width_px > 0 or self.avg_hip_width_px > 0):
            avg_width = (self.avg_shoulder_width_px + self.avg_hip_width_px) / 2
            volume_proxy = (self.avg_height_px * avg_width) / 1000.0  # Normalized factor
            
            # These thresholds are heuristics based on pixel volume trends
            if volume_proxy < 12:
                self.predicted_weight_category = f"Slight / Slim (Proxy={volume_proxy:.1f})"
            elif volume_proxy < 25:
                self.predicted_weight_category = f"Average Build (Proxy={volume_proxy:.1f})"
            else:
                self.predicted_weight_category = f"Husky / Heavyset (Proxy={volume_proxy:.1f})"

    def print_report(self):
        """Print a structured forensic report to stdout."""
        sep = "=" * 55
        print(f"\n{sep}")
        print("   FORENSIC VISUAL ANALYSIS REPORT")
        print(sep)
        print(f"  Video            : {self.video_path}")
        print(f"  Frames Processed : {self.total_frames}  (Detections: {self.detection_frames})")
        print(f"  Processing Time  : {self.processing_time_s:.1f}s")

        print(f"\n[ PHYSICAL METRICS (pixel-space) ]")
        print(f"  Est. Height       : {self.avg_height_px:.1f} px")
        print(f"  Est. Stride       : {self.avg_stride_px:.1f} px")
        print(f"  Shoulder Width    : {self.avg_shoulder_width_px:.1f} px")
        print(f"  Hip Width         : {self.avg_hip_width_px:.1f} px")
        print(f"  Arm Length        : {self.avg_arm_length_px:.1f} px")
        print(f"  Leg Length        : {self.avg_leg_length_px:.1f} px")

        print(f"\n[ PREDICTIVE PROFILING ]")
        print(f"  Predicted Gender  : {self.predicted_gender}")
        print(f"  Gait / Walk Style : {self.predicted_gait_type}")
        print(f"  Speed Category    : {self.predicted_speed_category}")
        print(f"  Arm Swing         : {self.arm_swing_quality}")
        print(f"  Body Proportions  : {self.body_proportions}")
        print(f"  Posture           : {self.posture_description}")
        print(f"  Weight / Build    : {self.predicted_weight_category}")
        print(f"  Head Size         : {self.head_size_description}")

        print(f"\n[ APPEARANCE ]")
        rgb = self.dominant_clothing_rgb
        rgb_str = f"RGB({rgb[0]},{rgb[1]},{rgb[2]})" if rgb else "N/A"
        print(f"  Clothing Color    : {self.dominant_clothing_color_name}  {rgb_str}")

        print(f"\n[ ACCESSORIES DETECTED ]")
        if self.accessories_detected:
            for a in self.accessories_detected:
                print(f"  • {a}")
        else:
            print("  None detected")

        print(f"\n[ IDENTITY PREDICTION (LSTM) ]")
        print(f"  Predicted Person  : {self.predicted_identity}")
        if self.identity_confidence_pct > 0:
            print(f"  Confidence        : {self.identity_confidence_pct:.1f}%")

        print(f"{sep}\n")

    def save_json(self, path: str):
        """Save report as a JSON file."""
        d = asdict(self)
        # Convert tuple to list for JSON serialisation
        if d.get("dominant_clothing_rgb") is not None:
            d["dominant_clothing_rgb"] = list(d["dominant_clothing_rgb"])
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"[ForensicReport] JSON saved → {path}")


# ---------------------------------------------------------------------------
# Video annotation helpers
# ---------------------------------------------------------------------------

_SIDEBAR_W = 340  # pixels

def _draw_sidebar(canvas: np.ndarray, report_snapshot: dict, frame_idx: int, fps: float):
    """Draw a dark sidebar panel with live forensic stats."""
    h, w = canvas.shape[:2]
    # Clear sidebar area
    canvas[:, w - _SIDEBAR_W:] = (15, 15, 15)

    def put(text, row, color=(200, 220, 200)):
        cv2.putText(canvas, text,
                    (w - _SIDEBAR_W + 8, 20 + row * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    put("=== FORENSIC ANALYSIS ===", 0, (80, 220, 120))
    put(f"Frame: {frame_idx}  FPS: {fps:.1f}", 1, (140, 140, 140))

    r = report_snapshot
    put(f"Height   : {r.get('avg_height_px', 0):.0f} px", 3)
    put(f"Stride   : {r.get('avg_stride_px', 0):.0f} px", 4)
    put(f"Shoulder : {r.get('avg_shoulder_width_px', 0):.0f} px", 5)
    put(f"Hip      : {r.get('avg_hip_width_px', 0):.0f} px", 6)
    put(f"Arm Len  : {r.get('avg_arm_length_px', 0):.0f} px", 7)
    put(f"Leg Len  : {r.get('avg_leg_length_px', 0):.0f} px", 8)

    # Posture indicator
    angle = abs(r.get('avg_posture_angle_deg', 0))
    p_color = (0, 255, 0) if angle < 5 else ((0, 215, 255) if angle < 15 else (0, 60, 255))
    put(f"Posture  : {r.get('posture_description', 'N/A')}", 10, p_color)
    put(f"Weight   : {r.get('predicted_weight_category', 'N/A')}", 11)
    put(f"Speed    : {r.get('predicted_speed_category', 'N/A')}", 12)
    put(f"Arm Swing: {r.get('arm_swing_quality', 'N/A')}", 13)

    put("--- Appearance ---", 14, (100, 180, 255))
    put(f"Clothing : {r.get('dominant_clothing_color_name', 'N/A')}", 15)

    accs = r.get("accessories_detected", [])
    put("--- Accessories ---", 17, (100, 180, 255))
    if accs:
        for i, a in enumerate(accs[:4]):
            put(f"  * {a}", 18 + i)
    else:
        put("  None", 18)

    put("--- Identity ---", 23, (100, 180, 255))
    put(f"Person: {r.get('predicted_identity', 'N/A')}", 24, (80, 255, 200))
    conf = r.get("identity_confidence_pct", 0)
    if conf > 0:
        put(f"Conf..: {conf:.1f}%", 25)


def _draw_posture_arc(frame: np.ndarray, kpts, top_left_x: int, top_left_y: int):
    """Draw a small posture arc on the frame near the person."""
    sh_mid = _midpoint(kpts, KP["l_shoulder"], KP["r_shoulder"])
    hp_mid = _midpoint(kpts, KP["l_hip"],      KP["r_hip"])
    if sh_mid is None or hp_mid is None:
        return
    angle = _estimate_posture(kpts)
    color = (0, 255, 0) if abs(angle) < 5 else ((0, 215, 255) if abs(angle) < 15 else (0, 60, 255))
    # Draw torso line
    cv2.line(frame,
             (int(sh_mid[0]), int(sh_mid[1])),
             (int(hp_mid[0]), int(hp_mid[1])),
             color, 2)
    # Label
    mid_x = int((sh_mid[0] + hp_mid[0]) / 2) + 5
    mid_y = int((sh_mid[1] + hp_mid[1]) / 2)
    cv2.putText(frame, f"{angle:.1f}deg", (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_video(
    video_path: str,
    output_video_path: Optional[str] = None,
    model_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> ForensicReport:
    """
    Full forensic analysis of a video file.

    Parameters
    ----------
    video_path        : Path to input video.
    output_video_path : If set, save annotated video here.
    model_path        : Path to trained LSTM .pth file (for identity).
    data_dir          : Processed dataset dir (for label mapping in LSTM).

    Returns
    -------
    ForensicReport with all fields populated.
    """
    t0 = time.time()
    report = ForensicReport(video_path=video_path)

    pose_model, obj_model = _get_models()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return report

    fps_in   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w    = frame_w + _SIDEBAR_W

    # Video writer
    writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps_in, (out_w, frame_h))

    # Frame-level buffers
    heights, strides, shoulders, hips, arms, legs = [], [], [], [], [], []
    postures, head_ratios, arm_asyms = [], [], []
    step_velocities = []
    color_samples = []
    acc_counts = {name: 0 for name in ACCESSORY_CLASSES.values()}
    gait_features_all = []  # For LSTM: (N, 62) accumulator

    prev_ankle_mid = None
    frame_idx = 0

    # Live snapshot dict for sidebar
    live_snapshot = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break
        frame_idx += 1
        report.total_frames += 1

        # ---- Pose inference ----
        pose_results = pose_model(frame, verbose=False)
        pr = pose_results[0]

        # ---- Object detection ----
        obj_results = obj_model(frame, verbose=False)
        or_ = obj_results[0]

        # ---- Accessories (always) ----
        frame_accessories = set()
        if or_.boxes is not None:
            for box in or_.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in ACCESSORY_CLASSES:
                    name = ACCESSORY_CLASSES[cls_id]
                    acc_counts[name] += 1
                    frame_accessories.add(name)

        # ---- Annotation canvas ----
        if writer:
            # Use YOLO's built-in rendered frame for skeleton
            annotated = pr.plot(kpt_radius=3, line_width=2)
            canvas = np.zeros((frame_h, out_w, 3), dtype=np.uint8)
            canvas[:, :frame_w] = annotated

        # ---- Per-person pose analysis (take highest-confidence person) ----
        kpts = None
        bbox = None
        if pr.keypoints is not None and len(pr.keypoints.data) > 0:
            best_idx = 0
            if pr.boxes is not None and len(pr.boxes.conf) > 0:
                best_idx = int(pr.boxes.conf.argmax().item())
            kpts = pr.keypoints.data[best_idx].cpu().numpy()
            if pr.boxes is not None and len(pr.boxes.xyxy) > 0:
                bbox = pr.boxes.xyxy[best_idx].cpu().numpy()

        if kpts is not None:
            report.detection_frames += 1

            # Height
            ankles_y = [kpts[KP["l_ankle"]][1], kpts[KP["r_ankle"]][1]]
            valid_ay = [y for y, c in zip(ankles_y, [kpts[KP["l_ankle"]][2], kpts[KP["r_ankle"]][2]]) if c > CONF_THRESH]
            h_px = 0.0
            if valid_ay and kpts[KP["nose"]][2] > CONF_THRESH:
                h_px = abs(max(valid_ay) - kpts[KP["nose"]][1])
            heights.append(h_px)

            # Stride, shoulder, hip, arm, leg
            st  = _kp_dist(kpts, KP["l_ankle"], KP["r_ankle"])
            sw  = _kp_dist(kpts, KP["l_shoulder"], KP["r_shoulder"])
            hw  = _kp_dist(kpts, KP["l_hip"], KP["r_hip"])
            la  = _kp_dist(kpts, KP["l_shoulder"], KP["l_elbow"]) + _kp_dist(kpts, KP["l_elbow"], KP["l_wrist"])
            ra  = _kp_dist(kpts, KP["r_shoulder"], KP["r_elbow"]) + _kp_dist(kpts, KP["r_elbow"], KP["r_wrist"])
            ll  = _kp_dist(kpts, KP["l_hip"], KP["l_knee"]) + _kp_dist(kpts, KP["l_knee"], KP["l_ankle"])
            rl  = _kp_dist(kpts, KP["r_hip"], KP["r_knee"]) + _kp_dist(kpts, KP["r_knee"], KP["r_ankle"])
            arm = max(la, ra)
            leg = max(ll, rl)

            strides.append(st)
            shoulders.append(sw)
            hips.append(hw)
            arms.append(arm)
            legs.append(leg)

            # Posture
            postures.append(_estimate_posture(kpts))

            # Head ratio
            head_ratios.append(_estimate_head_ratio(kpts, h_px))

            # Arm swing asymmetry
            arm_asyms.append(_arm_swing_asymmetry(kpts))

            # Step velocity
            curr_ankle = _ankle_midpoint(kpts)
            if curr_ankle is not None and prev_ankle_mid is not None:
                dist = float(np.linalg.norm(curr_ankle - prev_ankle_mid))
                velocity = dist * fps_in  # px per second
                step_velocities.append(velocity)
            prev_ankle_mid = curr_ankle

            # Clothing colour
            if bbox is not None:
                c = _get_dominant_color(frame, bbox)
                if c is not None:
                    color_samples.append(c)

            # Collect 62-dim gait feature for LSTM
            frame_feat = []
            for lm in kpts:
                frame_feat.extend([float(lm[0]), float(lm[1]), float(lm[2])])
            frame_feat.extend([h_px, st, sw, hw, arm, leg])
            # accessories (5 binary)
            frame_feat.extend([
                1.0 if "Backpack" in frame_accessories else 0.0,
                1.0 if "Umbrella" in frame_accessories else 0.0,
                1.0 if "Handbag"  in frame_accessories else 0.0,
                1.0 if "Tie"      in frame_accessories else 0.0,
                1.0 if "Suitcase" in frame_accessories else 0.0,
            ])
            # forensic behavioural (4)
            p_ang = postures[-1] if postures else 0.0
            h_rat = head_ratios[-1] if head_ratios else 0.0
            s_vel = step_velocities[-1] if step_velocities else 0.0
            a_asy = arm_asyms[-1] if arm_asyms else 0.0
            frame_feat.extend([p_ang, h_rat, s_vel, a_asy])
            
            # clothing colour (3)
            c_rgb = color_samples[-1] if color_samples else (0.0, 0.0, 0.0)
            frame_feat.extend([float(c_rgb[0]), float(c_rgb[1]), float(c_rgb[2])])

            gait_features_all.append(frame_feat)

            # Draw posture arc
            if writer:
                _draw_posture_arc(canvas, kpts, 0, 0)

                # Clothing colour swatch
                if color_samples:
                    avg_rgb = tuple(int(x) for x in np.mean(color_samples, axis=0))
                    bgr = (avg_rgb[2], avg_rgb[1], avg_rgb[0])
                    cv2.rectangle(canvas, (frame_w - 70, frame_h - 30),
                                  (frame_w - 10, frame_h - 10), bgr, -1)
                    cv2.putText(canvas, "Cloth", (frame_w - 70, frame_h - 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        else:
            gait_features_all.append([0.0] * 69)

        # Update live snapshot for sidebar
        if writer:
            live_snapshot.update({
                "avg_height_px": float(np.mean(heights)) if heights else 0,
                "avg_stride_px": float(np.mean(strides)) if strides else 0,
                "avg_shoulder_width_px": float(np.mean(shoulders)) if shoulders else 0,
                "avg_hip_width_px": float(np.mean(hips)) if hips else 0,
                "avg_arm_length_px": float(np.mean(arms)) if arms else 0,
                "avg_leg_length_px": float(np.mean(legs)) if legs else 0,
                "avg_posture_angle_deg": float(np.mean(postures)) if postures else 0,
                "posture_description": report.posture_description,
                "avg_arm_swing_asymmetry_deg": float(np.mean(arm_asyms)) if arm_asyms else 0,
                "arm_swing_quality": report.arm_swing_quality,
                "avg_step_velocity_px_per_s": float(np.mean(step_velocities)) if step_velocities else 0,
                "predicted_speed_category": report.predicted_speed_category,
                "dominant_clothing_color_name": _color_name(
                    tuple(int(x) for x in np.mean(color_samples, axis=0)) if color_samples else None
                ),
                "accessories_detected": [k for k, v in acc_counts.items() if v > 0],
                "predicted_identity": report.predicted_identity,
                "identity_confidence_pct": report.identity_confidence_pct,
            })
            _draw_sidebar(canvas, live_snapshot, frame_idx, fps_in)
            writer.write(canvas)

    cap.release()
    if writer:
        writer.release()
        print(f"[Video] Annotated output saved → {output_video_path}")

    # ---- Aggregate metrics ----
    def safe_mean(lst):
        valid = [v for v in lst if v > 0]
        return float(np.mean(valid)) if valid else 0.0

    report.avg_height_px           = safe_mean(heights)
    report.avg_stride_px           = safe_mean(strides)
    report.avg_shoulder_width_px   = safe_mean(shoulders)
    report.avg_hip_width_px        = safe_mean(hips)
    report.avg_arm_length_px       = safe_mean(arms)
    report.avg_leg_length_px       = safe_mean(legs)
    report.avg_posture_angle_deg   = float(np.mean(postures)) if postures else 0.0
    report.avg_head_ratio          = safe_mean(head_ratios)
    report.avg_arm_swing_asymmetry_deg = float(np.mean(arm_asyms)) if arm_asyms else 0.0
    report.avg_step_velocity_px_per_s  = safe_mean(step_velocities)

    # Clothing colour — most common cluster mean across sampled frames
    if color_samples:
        arr = np.array(color_samples, dtype=np.float32)
        report.dominant_clothing_rgb = tuple(int(x) for x in np.mean(arr, axis=0))

    # Accessories: present if detected in > 5% of all frames
    threshold = max(1, 0.05 * report.total_frames)
    report.accessories_detected = [k for k, v in acc_counts.items() if v >= threshold]

    # ---- LSTM identity prediction ----
    if model_path and os.path.exists(model_path) and gait_features_all:
        _run_lstm_prediction(report, gait_features_all, model_path, data_dir)

    # ---- Derive human-readable labels ----
    report._derive_attributes()

    report.processing_time_s = time.time() - t0
    return report


# ---------------------------------------------------------------------------
# LSTM identity helper
# ---------------------------------------------------------------------------

def _run_lstm_prediction(report: ForensicReport, gait_features_all, model_path, data_dir):
    """Run the trained ResNet model and populate identity fields in report."""
    try:
        import torch
        from model import GaitLSTM

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Label map
        label_map = {}
        if data_dir and os.path.isdir(data_dir):
            folders = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
            label_map = {i: name for i, name in enumerate(folders)}
        num_classes = max(2, len(label_map))

        # 1. Initialize Model (ResNet-1D)
        model = GaitLSTM(input_size=69, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # 2. Reshape and Normalize Features
        seq_len = 60
        feat = np.array(gait_features_all, dtype=np.float32)
        padded = np.zeros((seq_len, 69), dtype=np.float32)
        length = min(len(feat), seq_len)
        padded[:length] = feat[:length]
        
        # Apply Scaler if available (Crucial for ResNet accuracy)
        scaler_path = os.path.join(os.path.dirname(model_path), "scaler.pth")
        if os.path.exists(scaler_path):
            s_data = torch.load(scaler_path, map_location="cpu")
            mean = s_data['mean'].numpy()
            std = s_data['std'].numpy()
            padded = (padded - mean) / std

        tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 3. Inference
        with torch.no_grad():
            id_out, g_out = model(tensor)
            id_probs = torch.nn.functional.softmax(id_out, dim=1)[0]
            g_probs  = torch.nn.functional.softmax(g_out, dim=1)[0]
            
            idx = torch.argmax(id_probs).item()
            conf = float(id_probs[idx].item()) * 100
            
            g_idx = torch.argmax(g_probs).item()
            gender = "Male" if g_idx == 0 else "Female"

        report.predicted_identity = label_map.get(idx, f"Suspect_{idx}")
        report.identity_confidence_pct = conf
        report.predicted_gender = gender

    except Exception as e:
        import traceback
        print(f"[LSTM] Identity prediction failed: {e}")
        traceback.print_exc()
        report.predicted_identity = f"Model error: {str(e)[:50]}"


# ---------------------------------------------------------------------------
# CLI entry point (optional standalone use)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visual Forensics Engine")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--output_video", default=None, help="Annotated output video path")
    parser.add_argument("--report_json",  default=None, help="JSON report output path")
    parser.add_argument("--model",    default=os.path.join("models", "best_gait_model.pth"))
    parser.add_argument("--data_dir", default=os.path.join("dataset", "processed"))
    args = parser.parse_args()

    rpt = analyze_video(
        video_path=args.video,
        output_video_path=args.output_video,
        model_path=args.model,
        data_dir=args.data_dir,
    )
    rpt.print_report()
    if args.report_json:
        rpt.save_json(args.report_json)
