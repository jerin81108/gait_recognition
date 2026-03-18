import os
import math
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# Load YOLOv8 models (pose and standard object detection)
pose_model = YOLO("yolov8n-pose.pt")
obj_model = YOLO("yolov8n.pt")  # Standard YOLOv8 for COCO objects

# YOLOv8 pose model provides 17 keypoints
LANDMARK_COUNT = 17
VALUES_PER_LANDMARK = 3  # x, y, confidence
POSE_DIM = LANDMARK_COUNT * VALUES_PER_LANDMARK  # 51

# Forensic Derived Features:
#  6 pose metrics   : Height, Stride, Shoulder W, Hip W, Arm Len, Leg Len
#  5 accessories    : Backpack(24), Umbrella(25), Handbag(26), Tie(27), Suitcase(28)
#  4 behavioural    : Posture angle, Head ratio, Step velocity, Arm-swing asymmetry
#  3 clothing colour: Dominant R, G, B
FORENSIC_DIM = 6 + 5 + 4 + 3   # = 18
FEATURE_DIM = POSE_DIM + FORENSIC_DIM  # = 69


def extract_gait_features(video_path: str, max_frames: int = None) -> np.ndarray | None:
    """
    Extracts 2D pose landmarks and forensic details from a video using Ultralytics YOLOv8.

    Returns an N x 69 array containing:
      [0:51]  Pose features   — 17 landmarks × 3 (x, y, confidence)
      [51:57] Pose metrics    — Height, Stride, Shoulder W, Hip W, Arm Len, Leg Len
      [57:62] Accessories     — Backpack, Umbrella, Handbag, Tie, Suitcase (binary)
      [62]    Posture angle   — torso lean degrees from vertical
      [63]    Head ratio      — head height / body height
      [64]    Step velocity   — inter-frame ankle displacement (pixels)
      [65]    Arm asymmetry   — abs difference in left/right arm angles (degrees)
      [66:69] Clothing colour — dominant torso colour R, G, B
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: could not open video '{video_path}'")
        return None

    features = []
    detected_count = 0
    _prev_ankle_mid = [None]  # mutable container for inter-frame step velocity

    # COCO Class IDs for forensic accessories
    ACCESSORY_CLASSES = {24: 0, 25: 1, 26: 2, 27: 3, 28: 4}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames is not None and len(features) >= max_frames:
            break

        # Run YOLO inference
        pose_results = pose_model(frame, verbose=False)
        pose_result = pose_results[0]
        
        obj_results = obj_model(frame, verbose=False)
        obj_result = obj_results[0]

        if pose_result.keypoints is not None and len(pose_result.keypoints.data) > 0:
            # 1. Base Pose Keypoints (51 dimensions)
            kpts = pose_result.keypoints.data[0].cpu().numpy()
            
            frame_features = []
            for landmark in kpts:
                # landmark is [x, y, conf]
                frame_features.extend([landmark[0], landmark[1], landmark[2]])
                
            # 2. Derived Pose Metrics (6 dimensions)
            # Helper to calculate distance
            def p_dist(p1, p2):
                if p1[2] > 0.3 and p2[2] > 0.3: # confidence threshold
                    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                return 0.0

            # COCO Keypoints: 0 N, 1 LE, 2 RE, 3 LEar, 4 REar, 5 LS, 6 RS, 7 LE, 8 RE, 9 LW, 10 RW, 11 LH, 12 RH, 13 LK, 14 RK, 15 LA, 16 RA
            
            # Estimated Height (dist between nose and avg ankle)
            ankles_y = [kpts[15][1], kpts[16][1]]
            valid_ankles_y = [y for y, conf in zip(ankles_y, [kpts[15][2], kpts[16][2]]) if conf > 0.3]
            height = 0.0
            if valid_ankles_y and kpts[0][2] > 0.3:
                height = abs(max(valid_ankles_y) - kpts[0][1])
                
            # Stride (distance between ankles)
            stride = p_dist(kpts[15], kpts[16])
            
            # Shoulder and Hip Widths
            shoulder_width = p_dist(kpts[5], kpts[6])
            hip_width = p_dist(kpts[11], kpts[12])
            
            # Arm Length (avg of left and right if valid)
            l_arm = p_dist(kpts[5], kpts[7]) + p_dist(kpts[7], kpts[9])
            r_arm = p_dist(kpts[6], kpts[8]) + p_dist(kpts[8], kpts[10])
            arm_len = max(l_arm, r_arm)
            
            # Leg Length (avg of left and right if valid)
            l_leg = p_dist(kpts[11], kpts[13]) + p_dist(kpts[13], kpts[15])
            r_leg = p_dist(kpts[12], kpts[14]) + p_dist(kpts[14], kpts[16])
            leg_len = max(l_leg, r_leg)
            
            frame_features.extend([height, stride, shoulder_width, hip_width, arm_len, leg_len])
            
            # 3. Object Detection Indicators (5 dimensions)
            accessories = [0.0] * 5
            if obj_result.boxes is not None:
                for box in obj_result.boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id in ACCESSORY_CLASSES:
                        idx = ACCESSORY_CLASSES[cls_id]
                        accessories[idx] = 1.0
                        
            frame_features.extend(accessories)

            # 4. Posture angle (1 dimension)
            sh_mid_x = (kpts[5][0] + kpts[6][0]) / 2 if kpts[5][2] > 0.3 and kpts[6][2] > 0.3 else None
            sh_mid_y = (kpts[5][1] + kpts[6][1]) / 2 if kpts[5][2] > 0.3 and kpts[6][2] > 0.3 else None
            hp_mid_x = (kpts[11][0] + kpts[12][0]) / 2 if kpts[11][2] > 0.3 and kpts[12][2] > 0.3 else None
            hp_mid_y = (kpts[11][1] + kpts[12][1]) / 2 if kpts[11][2] > 0.3 and kpts[12][2] > 0.3 else None
            posture_angle = 0.0
            if None not in (sh_mid_x, sh_mid_y, hp_mid_x, hp_mid_y):
                dx = sh_mid_x - hp_mid_x
                dy = hp_mid_y - sh_mid_y
                if dy != 0:
                    posture_angle = math.degrees(math.atan(dx / dy))
            frame_features.append(posture_angle)

            # 5. Head-to-body ratio (1 dimension)
            head_ratio = 0.0
            nose = kpts[0]
            ears = [kpts[3], kpts[4]]
            valid_ears = [e for e in ears if e[2] > 0.3]
            if valid_ears and nose[2] > 0.3 and height > 0:
                avg_ear_y = float(np.mean([e[1] for e in valid_ears]))
                head_h = abs(float(nose[1]) - avg_ear_y) * 2.5
                head_ratio = head_h / height
            frame_features.append(head_ratio)

            # 6. Step velocity (inter-frame ankle displacement, 1 dimension)
            curr_ankles = []
            for ki in [15, 16]:
                if kpts[ki][2] > 0.3:
                    curr_ankles.append(kpts[ki][:2])
            step_vel = 0.0
            if curr_ankles and _prev_ankle_mid[0] is not None:
                curr_mid = np.mean(curr_ankles, axis=0)
                step_vel = float(np.linalg.norm(curr_mid - _prev_ankle_mid[0]))
            _prev_ankle_mid[0] = np.mean(curr_ankles, axis=0) if curr_ankles else None
            frame_features.append(step_vel)

            # 7. Arm-swing asymmetry (1 dimension)
            def _arm_angle(sh_i, el_i):
                if kpts[sh_i][2] > 0.3 and kpts[el_i][2] > 0.3:
                    return math.degrees(math.atan2(
                        float(kpts[el_i][1] - kpts[sh_i][1]),
                        float(kpts[el_i][0] - kpts[sh_i][0])
                    ))
                return None
            la = _arm_angle(5, 7)
            ra = _arm_angle(6, 8)
            arm_asym = abs(la - ra) if la is not None and ra is not None else 0.0
            frame_features.append(arm_asym)

            # 8. Clothing dominant colour (3 dimensions: R, G, B)
            clothing_rgb = [0.0, 0.0, 0.0]
            if obj_result.boxes is not None and len(obj_result.boxes.xyxy) > 0:
                # Use the largest detected person bounding box from pose model
                if pose_result.boxes is not None and len(pose_result.boxes.xyxy) > 0:
                    bb = pose_result.boxes.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
                    bh = y2 - y1
                    ty1 = y1 + int(bh * 0.30)
                    ty2 = y1 + int(bh * 0.70)
                    crop = frame[ty1:ty2, x1:x2]
                    if crop.size > 0:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        pixels = crop_rgb.reshape(-1, 3).astype(np.float32)
                        if len(pixels) >= 3:
                            k = min(3, len(pixels))
                            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                            _, labels_k, centers_k = cv2.kmeans(
                                pixels, k, None, crit, 3, cv2.KMEANS_RANDOM_CENTERS)
                            counts_k = np.bincount(labels_k.flatten())
                            dom = centers_k[np.argmax(counts_k)]
                            clothing_rgb = [float(dom[0]), float(dom[1]), float(dom[2])]
            frame_features.extend(clothing_rgb)

            features.append(frame_features)
            detected_count += 1
        else:
            _prev_ankle_mid[0] = None
            # Zero-pad missing frames so sequence length stays consistent
            features.append([0.0] * FEATURE_DIM)

    cap.release()

    if detected_count == 0:
        print(f"  Warning: no pose detected in any frame of '{video_path}'")
        return None

    return np.array(features, dtype=np.float32)


def main():
    raw_dir = os.path.join("dataset", "raw")
    processed_dir = os.path.join("dataset", "processed")

    if not os.path.exists(raw_dir):
        print(f"Raw dataset directory '{raw_dir}' does not exist.")
        return

    people_folders = [
        d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))
    ]

    if not people_folders:
        print(
            f"No person folders found in '{raw_dir}'. "
            "Expected structure: dataset/raw/person_A/video.mp4"
        )
        return

    print(f"Found {len(people_folders)} people in the dataset.")

    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    for person in people_folders:
        person_raw_dir = os.path.join(raw_dir, person)
        person_processed_dir = os.path.join(processed_dir, person)
        os.makedirs(person_processed_dir, exist_ok=True)

        videos = [
            f for f in os.listdir(person_raw_dir)
            if os.path.splitext(f)[1].lower() in video_extensions
        ]

        if not videos:
            print(f"No videos found for '{person}', skipping.")
            continue

        print(f"\nProcessing {len(videos)} video(s) for '{person}'...")

        for i, video in enumerate(tqdm(videos, desc=person, unit="video"), start=1):
            video_path = os.path.join(person_raw_dir, video)
            video_name = os.path.splitext(video)[0]
            save_path = os.path.join(person_processed_dir, f"{video_name}.npy")

            if os.path.exists(save_path):
                tqdm.write(f"  [{i}/{len(videos)}] Skipping (already processed): {video}")
                continue

            tqdm.write(f"  [{i}/{len(videos)}] Processing: {video}")
            features = extract_gait_features(video_path)

            if features is not None:
                np.save(save_path, features)
                tqdm.write(f"  Saved {features.shape} → {save_path}")
            else:
                tqdm.write(f"  Skipped (no valid poses): {video_path}")


if __name__ == "__main__":
    main()