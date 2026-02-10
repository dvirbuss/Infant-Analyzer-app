from pathlib import Path
import csv
import cv2
import numpy as np
import config
from .helpers import extract_keypoints_xy, normalize_coordinates, calculate_angle, draw_angle_arc  # if you keep it

def infer_video_with_angles(model, video_path: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = str(video_path)
    name = Path(video_path).stem

    out_video = out_dir / f"PE_trained_{name}.mp4"
    kp_tsv    = out_dir / f"Keypoints_trained_{name}.tsv"
    ang_tsv   = out_dir / f"Angles_trained_{name}.tsv"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    fw, fh = config.FIXED_RESOLUTION
    out = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

    frame_number = 0
    with open(kp_tsv, "w", newline="") as kpf, open(ang_tsv, "w", newline="") as anf:
        kp_writer = csv.writer(kpf, delimiter="\t")
        ang_writer = csv.writer(anf, delimiter="\t")

        kp_writer.writerow(["Frame", "Time"] + [f"{cat}_{axis}" for cat in config.CATEGORIES for axis in ["X","Y"]])
        ang_writer.writerow(["Frame", "Time", "R_Eye-Ear-Vertical", "R_Wrist-Elbow-Shoulder", "R_Hip-Knee-Ankle"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (fw, fh))
            frame_number += 1
            t = frame_number / fps

            results = model(frame)
            if results and results[0].keypoints is not None:
                kps_xy = extract_keypoints_xy(results[0].keypoints)
            else:
                kps_xy = [[0,0]] * len(config.CATEGORIES)

            normalized = [normalize_coordinates(x, y, fw, fh) for x, y in kps_xy]
            flat = [v for xy in normalized for v in xy]

            # angles (use ORIGINAL pixel coords for geometry)
            def get(name):
                return kps_xy[config.CATEGORIES.index(name)]

            r_shoulder = get("R_Shoulder")
            r_elbow    = get("R_Elbow")
            r_wrist    = get("R_Wrist")
            r_ear      = get("R_Ear")
            r_eye      = get("R_Eye")
            r_hip      = get("R_Hip")
            r_knee     = get("R_Knee")
            r_ankle    = get("R_Ankle")

            vertical_sp = [r_ear[0], r_ear[1] + 100]  # pixel vertical ref
            a_neck  = calculate_angle(vertical_sp, r_ear, r_eye)
            a_elbow = calculate_angle(r_wrist, r_elbow, r_shoulder)
            a_knee  = calculate_angle(r_hip, r_knee, r_ankle)

            kp_writer.writerow([frame_number, round(t, 3)] + flat)
            ang_writer.writerow([frame_number, round(t, 3), a_neck, a_elbow, a_knee])

            annotated = results[0].plot() if results else frame
            out.write(annotated)

    cap.release()
    out.release()

    return {
        "video_out": str(out_video),
        "keypoints_tsv": str(kp_tsv),
        "angles_tsv": str(ang_tsv),
    }
