import os
import cv2
from ultralytics import YOLO
from .helpers import extract_keypoints_xy, normalize_coordinates
import config

def is_head_facing_right(keypoints):
    nose = keypoints.get("Nose")
    r_ear = keypoints.get("R_Ear")
    l_ear = keypoints.get("L_Ear")
    if nose is None or r_ear is None or l_ear is None:
        return None
    if float(nose[0]) < float(l_ear[0]):
        return False  # looking right → need mirror
    if float(nose[0]) > float(r_ear[0]):
        return True   # looking left → ok
    return None

def mirror_video(input_path, output_path):
    flag = False
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if input_path == output_path:
        output_path = output_path[:-4] + "_tmp" + output_path[-4:]
        flag = True

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(cv2.flip(frame, 1))
    cap.release()
    out.release()

    if flag:
        os.remove(input_path)
        os.rename(output_path, input_path)

def save_first_frame_keypoints(model, video_path: str):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read first frame")

    results = model(frame)
    if results and results[0].keypoints is not None:
        keypoints_xy = extract_keypoints_xy(results[0].keypoints)
    else:
        keypoints_xy = [[0, 0]] * len(config.CATEGORIES)

    fw, fh = config.FIXED_RESOLUTION
    keypoints_xy = [normalize_coordinates(kp[0], kp[1], fw, fh) for kp in keypoints_xy]

    kp_dict = {
        "Nose": keypoints_xy[config.CATEGORIES.index("Nose")],
        "R_Ear": keypoints_xy[config.CATEGORIES.index("R_Ear")],
        "L_Ear": keypoints_xy[config.CATEGORIES.index("L_Ear")],
    }

    head_facing_right = is_head_facing_right(kp_dict)
    cap.release()

    if not head_facing_right:
        mirror_video(video_path, video_path)
