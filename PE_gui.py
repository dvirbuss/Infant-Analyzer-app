# -------------------------- Libraries --------------------------

import os
import cv2
import csv
import datetime
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.impute import KNNImputer
from tkinter import filedialog, messagebox
from tkcalendar import DateEntry
from tkinterdnd2 import TkinterDnD, DND_FILES

# -------------------------- GLOBAL SETTINGS --------------------------

FIXED_RESOLUTION = (640, 480)  # Frame resolution
CATEGORIES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"
]

# Global boolean score lists
PRONE_SCORE =  [True, True] + [False] * 9       # Total 11
SUPINE_SCORE = [True, True] + [False] * 6       # Total 8
SITTING_SCORE = [True] + [False] * 6            # Total 7
STANDING_SCORE = [True] + [False] * 2           # Total 3

# Global variable to track mouse position
mouse_position = "(0.00%, 0.00%)"


# ==================== Helper Functions ====================

# Callback function to display normalized coordinates of the mouse pointer
def mouse_callback(event, x, y, flags, param):
    global mouse_position
    if event == cv2.EVENT_MOUSEMOVE:
        frame_width, frame_height = param
        normalized_x, normalized_y = normalize_coordinates(x, y, frame_width, frame_height)
        mouse_position = f"({normalized_x:.2f}%, {normalized_y:.2f}%)"

# Normalize (x, y) coordinates to percentages of the frame
def normalize_coordinates(x, y, frame_width, frame_height):
    normalized_x = round((x / frame_width) * 100, 2)
    normalized_y = round((y / frame_height) * 100, 2)
    return normalized_x, normalized_y

# Extracts (x, y) keypoints from YOLO object
def extract_keypoints_xy(keypoints_object):
    if keypoints_object is None:
        return [[0, 0]] * len(CATEGORIES)  # Return zeros for all keypoints if none exist
    return keypoints_object.xy.cpu().numpy()[0]  # Extract the (x, y) keypoints

# Converts list of booleans to list of '1' and '0' strings
def to_str_list(bool_list):
    return ["1" if val else "0" for val in bool_list]

# Calculates angle formed by 3 points (in degrees)
def calculate_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ab = a - b
    cb = c - b
    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)
    if norm_ab == 0 or norm_cb == 0:
        return 0.0
    cosine_angle = np.dot(ab, cb) / (norm_ab * norm_cb)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return round(angle, 2)

# Draw an angle arc with degree value on the video frame
def draw_angle_arc(frame, p1, p2, p3, angle, color, radius=20, arc_text_offset=(10, -10)):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    thickness = 2
    start_angle = np.degrees(np.arctan2(a[1] - b[1], a[0] - b[0]))
    end_angle = np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0]))
    if start_angle > end_angle:
        start_angle, end_angle = end_angle, start_angle
    cv2.ellipse(
        frame,
        center=(int(b[0]), int(b[1])),
        axes=(radius, radius),
        angle=0,
        startAngle=start_angle,
        endAngle=end_angle,
        color=color,
        thickness=thickness,
    )
    arc_text_position = (int(b[0] + arc_text_offset[0]), int(b[1] + arc_text_offset[1]))
    cv2.putText(
        frame,
        f"{int(angle)}",
        arc_text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    # Define helper function for piecewise linear interpolation
def create_curve(x_points, y_points, months):
    return np.interp(months, x_points, y_points)

# Perform KNN imputation for missing keypoints in a TSV file
def knn_impute_keypoints_tsv(tsv_path, n_neighbors=5, threshold=0.6):
    df = pd.read_csv(tsv_path, delimiter="\t")
    df_copy = df.copy()
    keypoint_cols = df.columns[2:] #(excluding "Frame" and "Time")
    df_copy[keypoint_cols] = df_copy[keypoint_cols].replace(0, np.nan)
    missing_ratio = df_copy[keypoint_cols].isna().mean()

    for col in keypoint_cols:
        if missing_ratio[col] > threshold:
            print(f"Column {col} has more than {threshold * 100}% missing values. Setting all values to 0.")
            df_copy[col] = 0

    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_copy[keypoint_cols] = imputer.fit_transform(df_copy[keypoint_cols])
    df_copy[keypoint_cols] = df_copy[keypoint_cols].round(2)
    tsv_output = df_copy.to_csv(tsv_path, sep="\t", index=False)
    return tsv_output


###################### Algo Mirror video ###########
# --- Process the first frame of a video: extract keypoints and mirror if baby isn't facing right ---
def save_first_frame_keypoints(trained_model_path, video_path):
    # Check if model and video files exist
    if not os.path.exists(trained_model_path):
        raise FileNotFoundError(f"Trained model file not found: {trained_model_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")

    # Load the YOLO model
    print("Loading the trained model...")
    model = YOLO(trained_model_path)

    # Open the video and read the first frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    # Run pose estimation on the first frame
    print("Extracting keypoints from the first frame...")
    results = model(frame)
    if results and results[0].keypoints is not None:
        keypoints_xy = extract_keypoints_xy(results[0].keypoints)
    else:
        keypoints_xy = [[0, 0]] * len(CATEGORIES)

    # Normalize the detected keypoints
    frame_width, frame_height = FIXED_RESOLUTION
    keypoints_xy = [normalize_coordinates(kp[0], kp[1], frame_width, frame_height) for kp in keypoints_xy]

    # Extract key head points for orientation check
    keypoints_dict = {
        "Nose": keypoints_xy[CATEGORIES.index("Nose")],
        "R_Ear": keypoints_xy[CATEGORIES.index("R_Ear")],
        "L_Ear": keypoints_xy[CATEGORIES.index("L_Ear")]
    }

    # Determine orientation and mirror if needed
    head_facing_right = is_head_facing_right(keypoints_dict)
    if not head_facing_right:
        cap.release()
        mirror_video(video_path, video_path)
        print("Video successfully mirrored\n")
    else:
        print("Video is in correct format\n")

# --- Mirror a video horizontally and optionally overwrite the original if needed -
def mirror_video(input_path, output_path):
    flag = False
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if input_path == output_path:
        output_path = output_path[:-4] + "zzz" + output_path[-4:]
        flag = True
    # Define the codec and create VideoWriter
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame horizontally
        mirrored_frame = cv2.flip(frame, 1)

        # Write the mirrored frame to the output video
        out.write(mirrored_frame)

    cap.release()
    out.release()
    if flag:
        os.remove(input_path)
        os.rename(output_path,output_path[:-7] + output_path[-4:])
    print(f"Mirrored video saved to {output_path}")

# Determines if the head is facing right based on relative nose and ear positions
def is_head_facing_right(keypoints):
    nose = keypoints.get("Nose")
    r_ear = keypoints.get("R_Ear")
    l_ear = keypoints.get("L_Ear")
    print("\nNose X val", nose[0], "Right ear X val", r_ear[0])
    print("if Nose val > right ear ===> correct format")
    print("Nose X val", nose[0], "Left ear X val", l_ear[0])
    print("if Nose val < left ear ===> correct format\n")

    # Compare X-coordinates
    if float(nose[0]) < float(l_ear[0]):  # Nose X < Left Ear X → Looking Right
        return False
    elif float(nose[0]) > float(r_ear[0]):  # Nose X > Right Ear X → Looking Left
        return True
    return None  # Defth.splitault case if something is off

# ==================== Output Functions ====================

def plot_angle_changes_over_time(angles_tsv_path, save_path=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    df = pd.read_csv(angles_tsv_path, delimiter="\t")

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df["Frame"], df["R_Eye-Ear-Vertical"], label="R_Eye-Ear-Vertical", color="blue")
    ax1.set_title("Change of Neck Angle Over Time")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Angle (°)")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(df["Frame"], df["R_Wrist-Elbow-Shoulder"], label="R_Wrist-Elbow-Shoulder", color="green")
    ax2.set_title("Change of Elbow Angle Over Time")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Angle (°)")
    ax2.grid(True)
    ax2.legend()

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(df["Frame"], df["R_Hip-Knee-Ankle"], label="R_Hip-Knee-Ankle", color="purple")
    ax3.set_title("Change of Knee Angle Over Time")
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Angle (°)")
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


#Colors the score cells in the AIMS table: Green for score 1, red for score 0.
def color_aims_score_cells(table):
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_fontsize(10)
            cell.set_text_props(weight='bold')
            continue

        text = cell.get_text().get_text()
        if "(" in text and ")" in text:
            score_str = text.split("(")[-1].split(")")[0].strip()
            if score_str == "1":
                cell.get_text().set_color("green")
            elif score_str == "0":
                cell.get_text().set_color("red")

# Plot the full AIMS report with growth curves and developmental scores
def plot_aims_report(months, curves, baby_age_months, baby_score, report_data, save_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    # 3 rows: Plot (4), Title (0.3), Table (1)
    fig = plt.figure(figsize=(12, 13))
    gs = gridspec.GridSpec(3, 1, height_ratios=[4, 0.3, 1])

    # --- Plot: AIMS curve ---
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(months, curves["p90"], label="90th %ile", color='green')
    ax0.plot(months, curves["p75"], label="75th %ile", color='deepskyblue')
    ax0.plot(months, curves["p50"], label="50th %ile", color='gold')
    ax0.plot(months, curves["p25"], label="25th %ile", color='silver')
    ax0.plot(months, curves["p10"], label="10th %ile", color='darkorange')
    ax0.plot(months, curves["p5"], label="5th %ile", color='blue')
    ax0.scatter(baby_age_months, baby_score, color="black", s=40,
                label=f"Infant: {baby_age_months} months, Score: {baby_score}")
    ax0.axhline(y=4, color='gray', linewidth=1)

    ax0.set_xlim(0, 6)
    ax0.set_ylim(0, 30)
    ax0.set_xticks(np.arange(0, 7, 1))
    ax0.set_yticks(np.arange(0, 28, 2))
    ax0.set_xlabel("Age (months)")
    ax0.set_ylabel("AIMS Score")
    ax0.set_title("AIMS Score vs Age - Infant Development Position")
    ax0.grid(True)
    ax0.legend()

    # --- Title Row ---
    ax_title = fig.add_subplot(gs[1])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, "AIMS Report", ha="center", va="center", fontsize=14, fontweight='bold')

    # --- Table Row ---
    ax1 = fig.add_subplot(gs[2])
    ax1.axis("off")

    # Prepare table data
    max_rows = max(len(v) for v in report_data.values())
    columns = list(report_data.keys())
    table_data = []

    for i in range(max_rows):
        row = []
        for col in columns:
            if i < len(report_data[col]):
                label, score = report_data[col][i]
                row.append(f"{label} ({score})")
            else:
                row.append("")
        table_data.append(row)

    # Create table
    table = ax1.table(cellText=table_data, colLabels=columns, loc="center", cellLoc="left")
    table.scale(1, 2.5)
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Apply coloring
    color_aims_score_cells(table)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def generate_aims_report(total_scores):
    """
    Takes a list of lists of scores [prone, supine, sitting, standing]
    and maps them to the corresponding AIMS items.
    Each item will hold its real score (0 or 1).
    """
    prone   = to_str_list(total_scores[0])
    supine  = to_str_list(total_scores[1])
    sitting = to_str_list(total_scores[2])
    standing= to_str_list(total_scores[3])

    return {
        "Prone": [
            ("Prone lying 1", prone[0]), ("Prone lying 2", prone[1]),
            ("Prone prop", prone[2]), ("Forearm support 1", prone[3]),
            ("Prone mobility", prone[4]), ("Forearm support 2", prone[5]),
            ("Extended arm support", prone[6]), ("Rolling prone to supine without rotation", prone[7]),
            ("Swimming", prone[8]), ("Reaching from forearm support", prone[9]), ("Pivoting", prone[10])
        ],
        "Supine": [
            ("Supine lying 1", supine[0]), ("Supine lying 2", supine[1]), ("Supine lying 3", supine[2]),
            ("Supine lying 4", supine[3]), ("Hands to knees", supine[4]),
            ("Active extension", supine[5]), ("Hands to feet", supine[6]),
            ("Rotating supine to prone without rotation", supine[7])
        ],
        "Sitting": [
            ("Sitting with support", sitting[0]), ("Sitting with propped arms", sitting[1]),
            ("Pull to sit", sitting[2]), ("Unsustained sitting", sitting[3]),
            ("Sitting with arm support", sitting[4]),
            ("Unsustained sitting without arm support", sitting[5]),
            ("Weight shift in unsustained sitting", sitting[6])
        ],
        "Standing": [
            ("Support standing 1", standing[0]), ("Support standing 2", standing[1]), ("Support standing 3", standing[2])
        ]
    }

import matplotlib.pyplot as plt
import numpy as np

def plot_aims_percentile_bar(baby_age_months, baby_score, curves, save_path=None):
    # Interpolate Y-values at given age for all percentiles
    age_index = np.abs(np.linspace(0, 6, 100) - baby_age_months).argmin()
    cutoffs = {
        "5th":   curves["p5"][age_index],
        "10th":  curves["p10"][age_index],
        "25th":  curves["p25"][age_index],
        "50th":  curves["p50"][age_index],
        "75th":  curves["p75"][age_index],
        "90th":  curves["p90"][age_index],
    }

    # Compute boundaries for coloring
    boundaries = [
        (4, cutoffs["5th"], '<5th', 'red'),  # Very Low
        (cutoffs["5th"], cutoffs["10th"], '5–10th', 'orangered'),  # Low
        (cutoffs["10th"], cutoffs["25th"], '10–25th', 'gold'),  # Below Average
        (cutoffs["25th"], cutoffs["50th"], '25–50th', 'yellowgreen'),  # Neutral
        (cutoffs["50th"], cutoffs["75th"], '50–75th', 'mediumseagreen'),  # Above Avg
        (cutoffs["75th"], cutoffs["90th"], '75–90th', 'forestgreen'),  # High
        (cutoffs["90th"], cutoffs["90th"] + 3, '>90th', 'darkgreen')  # Very High
    ]

    # Define explanations per percentile range
    explanations = {
        '<5th': "This indicates significantly delayed motor development for this age.",
        '5–10th': "Motor development is below average; monitoring is advised.",
        '10–25th': "Motor skills are slightly below average; some delay may be present.",
        '25–50th': "Motor development is in the average range.",
        '50–75th': "The infant shows above-average motor development.",
        '75–90th': "Motor skills are well-developed for age.",
        '>90th': "The infant demonstrates advanced motor development for this age."
    }

    # Determine percentile range
    percentile_range = "Unknown"
    for start, end, label, _ in boundaries:
        if start <= baby_score < end:
            percentile_range = label
            break
    if baby_score >= cutoffs["90th"]:
        percentile_range = ">90th"
    elif baby_score < cutoffs["5th"]:
        percentile_range = "<5th"

    # Plot setup
    fig = plt.figure(figsize=(12, 3.5), dpi=100)
    ax = fig.add_axes([0.05, 0.4, 0.9, 0.45])  # Slightly higher to make space for arrow

    # Color bands and percentile labels
    for start, end, label, color in boundaries:
        ax.axvspan(start, end, color=color)
        ax.text((start + end) / 2, 0.85, label, ha='center', va='center',
                color='black', fontsize=10, weight='bold')

    # Arrow score marker below bar
    ax.plot(baby_score, -0.2, marker='^', color='black', markersize=18)

    # Axis lines and ticks
    xmin = 4
    xmax = int(cutoffs["90th"] + 3)
    for x in range(xmin, xmax + 1):
        ax.axvline(x, color='white', linestyle='--', linewidth=0.8, alpha=0.3)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.4, 1.1)  # <<< KEY LINE
    ax.set_xticks(range(xmin, xmax + 1))
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=9)
    ax.set_xlabel("AIMS Score")

    # Titles below
    ax.set_title(f"AIMS Score Percentile Range at {round(baby_age_months, 2)} Months", pad=10)
    fig.text(0.5, 0.08, f'The infant is in the "{percentile_range}" percentile',
             ha='center', va='center', fontsize=10, weight='bold')
    fig.text(0.5, 0.02, f"Explanation: {explanations.get(percentile_range, 'No explanation available.')}",
             ha='center', va='center', fontsize=10, style='italic')

    if save_path:
        plt.savefig(save_path)
    plt.show()


# ==================== Aims Test ====================

# Sums all True values from the global score lists and returns the integer total
def score_sum(keypoints_tsv_path, angles_tsv_path):
    prone_total(keypoints_tsv_path, angles_tsv_path)
    supine_total(keypoints_tsv_path, angles_tsv_path)
    sitting_total(keypoints_tsv_path, angles_tsv_path)
    standing_total(keypoints_tsv_path, angles_tsv_path)
    total = sum(map(int, PRONE_SCORE + SUPINE_SCORE + SITTING_SCORE + STANDING_SCORE))
    return total

######### 1. Prone pose #######

def prone_total(keypoints_tsv_path, angles_tsv_path):
    prone_prop(angles_tsv_path)                            # PRONE_SCORE[2]
    forearm_support_1(keypoints_tsv_path, angles_tsv_path) # PRONE_SCORE[3]
    #prone_mobility(angles_tsv_path)                        # PRONE_SCORE[4]
    #forearm_support_2(keypoints_tsv_path, angles_tsv_path) # PRONE_SCORE[5]

# Updates PRONE_SCORE[2] based on whether at least 90 consecutive frames show angle > 45° for R_Eye-Ear-Vertical
def prone_prop(angles_tsv_path):
    global PRONE_SCORE
    df = pd.read_csv(angles_tsv_path, delimiter="\t")

    if "R_Eye-Ear-Vertical" not in df.columns:
        print("R_Eye-Ear-Vertical column not found.")
        PRONE_SCORE[2] = False
        return

    above_45 = df["R_Eye-Ear-Vertical"] > 45
    count = 0
    max_count = 0
    for value in above_45:
        if value:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0

    PRONE_SCORE[2] = max_count >= 90

# Updates PRONE_SCORE[3] to True if PRONE_SCORE[2] is True and R_Elbow_X >= R_Shoulder_X in all frames
def forearm_support_1(keypoints_tsv_path, angles_tsv_path, fps=30):
    global PRONE_SCORE

    prone_prop(angles_tsv_path)  # Ensure PRONE_SCORE[2] is up to date
    if not PRONE_SCORE[2]:
        PRONE_SCORE[3] = False
        return

    df = pd.read_csv(keypoints_tsv_path, delimiter="\t")
    if "R_Elbow_X" not in df or "R_Shoulder_X" not in df:
        print("Missing elbow or shoulder X columns.")
        PRONE_SCORE[3] = False
        return

    # Check elbow behind or aligned with shoulder (static support posture)
    condition = df["R_Elbow_X"] >= df["R_Shoulder_X"]

    # Check for 3 seconds of consecutive frames
    count = 0
    max_count = 0
    for val in condition:
        count = count + 1 if val else 0
        max_count = max(max_count, count)

    PRONE_SCORE[3] = max_count >= 3 * fps


def prone_mobility(keypoints_tsv_path, angles_tsv_path, fps=30):
    global PRONE_SCORE

    if not PRONE_SCORE[2]:  # Must pass Prone Prop
        PRONE_SCORE[4] = False
        return

    df_kp = pd.read_csv(keypoints_tsv_path, delimiter="\t")
    df_ang = pd.read_csv(angles_tsv_path, delimiter="\t")

    required_cols = ["R_Elbow_Y", "R_Knee_Y"]
    if any(col not in df_kp.columns for col in required_cols) or "R_Eye-Ear-Vertical" not in df_ang:
        PRONE_SCORE[4] = False
        return

    # --- Step 2: Elbow above ground (using knee as floor proxy)
    # In image coordinates, "up" is smaller Y → so: Elbow_Y < Knee_Y - margin
    margin = 3  # percentage units
    elbow_off_ground = df_kp["R_Elbow_Y"] < (df_kp["R_Knee_Y"] - margin)

    # --- Step 3: Neck raised to 90°
    neck_angle_high = df_ang["R_Eye-Ear-Vertical"] > 85  # allow some tolerance

    # --- Combine conditions
    condition = elbow_off_ground & neck_angle_high

    # Check for 3 sec consecutive True values
    count = 0
    max_count = 0
    for val in condition:
        count = count + 1 if val else 0
        max_count = max(max_count, count)

    PRONE_SCORE[4] = max_count >= 3 * fps

def forearm_support_2(keypoints_tsv_path, angles_tsv_path, fps=30):
    global PRONE_SCORE

    if not PRONE_SCORE[3]:  # Must pass Forearm Support 1
        PRONE_SCORE[5] = False
        return

    df_kp = pd.read_csv(keypoints_tsv_path, delimiter="\t")
    df_ang = pd.read_csv(angles_tsv_path, delimiter="\t")

    if "R_Elbow_X" not in df_kp or "R_Shoulder_X" not in df_kp or "R_Eye-Ear-Vertical" not in df_ang:
        PRONE_SCORE[5] = False
        return

    # Frame-by-frame conditions
    elbow_forward = df_kp["R_Elbow_X"] > df_kp["R_Shoulder_X"]
    head_angle_high = df_ang["R_Eye-Ear-Vertical"] > 60
    condition_met = elbow_forward & head_angle_high

    # Check for 3 seconds = ~90 consecutive frames
    count = 0
    max_count = 0
    for val in condition_met:
        count = count + 1 if val else 0
        max_count = max(max_count, count)

    PRONE_SCORE[5] = max_count >= 3 * fps


######### 2. Supine pose #######

def supine_total(keypoints_tsv_path, angles_tsv_path):
    return

######### 3. Sitting pose #######

def sitting_total(keypoints_tsv_path, angles_tsv_path):
    return

######### 4. Standing pose #######

def standing_total(keypoints_tsv_path, angles_tsv_path):
    return

#####

def test_model_with_angles(trained_model_path, video_path, output_video_path, keypoints_tsv_path, angles_tsv_path, angles_plt_path, expert_plt_path, parent_plt_path):
    """
    Perform pose estimation on a video, calculate angles, and annotate them on the video.

    Args:
        trained_model_path: Path to the trained YOLO model.
        video_path: Path to the input video.
        output_video_path: Path to save the output video.
        keypoints_tsv_path: Path to save the keypoints data as a TSV file.
        angles_tsv_path: Path to save the angles data as a TSV file.

    Returns:
        None. Saves the output video with visualized pose estimation and separate TSV files.
    """
    if not os.path.exists(trained_model_path):
        raise FileNotFoundError(f"Trained model file not found: {trained_model_path}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")

    print("Loading the trained model...")
    model = YOLO(trained_model_path)  # Load the trained model

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width, frame_height = FIXED_RESOLUTION
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_number = 0
    with open(keypoints_tsv_path, "w", newline="") as keypoints_file, open(angles_tsv_path, "w", newline="") as angles_file:
        keypoints_writer = csv.writer(keypoints_file, delimiter="\t")
        angles_writer = csv.writer(angles_file, delimiter="\t")

        keypoints_writer.writerow(["Frame", "Time"] + [f"{cat}_{axis}" for cat in CATEGORIES for axis in ["X", "Y"]])
        angles_writer.writerow(["Frame", "Time", "R_Eye-Ear-Vertical", "R_Wrist-Elbow-Shoulder", "R_Hip-Knee-Ankle"])

        # Set up mouse callback
        cv2.namedWindow("Pose Estimation with Angles")
        cv2.setMouseCallback("Pose Estimation with Angles", mouse_callback, FIXED_RESOLUTION)

        print("Starting pose estimation on video...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to fixed resolution
            frame = cv2.resize(frame, FIXED_RESOLUTION)
            frame_number += 1
            print(f"Processing frame {frame_number}")
            time = frame_number / fps

            # Perform pose estimation
            results = model(frame)
            if len(results) > 0 and results[0].keypoints is not None:
                keypoints_xy = extract_keypoints_xy(results[0].keypoints)
            else:
                keypoints_xy = [[0, 0]] * len(CATEGORIES)  # Zero for missing keypoints

            original_keypoints_xy = keypoints_xy.copy()
            # Keep only right-side keypoints and nose for visualization
            for i, (x, y) in enumerate(keypoints_xy):
                if CATEGORIES[i] not in ["Nose", "R_Eye", "R_Ear", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hip",
                                         "R_Knee", "R_Ankle"]:
                    keypoints_xy[i] = [0, 0]  # Hide unwanted keypoints visually

            # Normalize keypoints
            normalized_keypoints = [normalize_coordinates(kp[0], kp[1], frame_width, frame_height) for kp in original_keypoints_xy]
            # Calculate angles

            r_shoulder = keypoints_xy[CATEGORIES.index("R_Shoulder")]
            r_elbow = keypoints_xy[CATEGORIES.index("R_Elbow")]
            r_wrist = keypoints_xy[CATEGORIES.index("R_Wrist")]
            r_ear = keypoints_xy[CATEGORIES.index("R_Ear")]
            r_eye = keypoints_xy[CATEGORIES.index("R_Eye")]
            vertical_sp = [r_ear[0], round(r_ear[1] + 100, 2)]
            r_hip = keypoints_xy[CATEGORIES.index("R_Hip")]
            r_knee = keypoints_xy[CATEGORIES.index("R_Knee")]
            r_ankle = keypoints_xy[CATEGORIES.index("R_Ankle")]

            angle_hip_knee_ankle = calculate_angle(r_hip, r_knee, r_ankle)
            angle_vertical_ear_eye = calculate_angle(vertical_sp, r_ear, r_eye)
            cv2.circle(frame, (int(vertical_sp[0]), int(vertical_sp[1])), 5, (0, 215, 255), -1)
            cv2.line(frame, (int(vertical_sp[0]), int(vertical_sp[1])), (int(r_ear[0]), int(r_ear[1])), (0, 215, 255),2)
            draw_angle_arc(frame, vertical_sp, r_ear, r_eye, angle_vertical_ear_eye, color=(0, 215, 255), radius=20, arc_text_offset=(15, -15))
            angle_wrist_elbow_shoulder = calculate_angle(r_wrist, r_elbow, r_shoulder)

            # Write keypoints and angles to their respective TSV files
            percentage_keypoints_flat = [val for sublist in normalized_keypoints for val in sublist]  # Flatten
            keypoints_writer.writerow([frame_number, round(time, 3)] + percentage_keypoints_flat)
            angles_writer.writerow([frame_number, round(time, 3), angle_vertical_ear_eye, angle_wrist_elbow_shoulder,
                                    angle_hip_knee_ankle])

            # Annotate pose and angles on the frame
            annotated_frame = results[0].plot() if len(results) > 0 else frame

            # Draw arcs and angles for R_Shoulder-Eye-Nose
            draw_angle_arc(annotated_frame, vertical_sp, r_ear, r_eye, angle_vertical_ear_eye,
                color=(255, 255, 255), radius=20, arc_text_offset=(15, -15))

            # Draw arcs and angles for R_Wrist-Elbow-Shoulder
            draw_angle_arc(annotated_frame, r_wrist, r_elbow, r_shoulder, angle_wrist_elbow_shoulder,
                           color=(255, 255, 255), radius=20, arc_text_offset=(15, -15))

            draw_angle_arc(annotated_frame, r_hip, r_knee, r_ankle, angle_hip_knee_ankle,
                           color=(255, 255, 255), radius=20, arc_text_offset=(15, -15))

            # Add mouse position and degree text above the frame
            cv2.putText(
                annotated_frame,
                f"Mouse: {mouse_position}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                f"Neck Angle: {angle_vertical_ear_eye} deg",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                f"Elbow Angle: {angle_wrist_elbow_shoulder} deg",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                annotated_frame,
                f"Knee Angle: {angle_hip_knee_ankle} deg",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            out.write(annotated_frame)  # Write the annotated frame to the output video

            # Display the frame with angles and mouse position
            cv2.imshow("Pose Estimation with Angles", annotated_frame)

            # Check for quit via 'q' or window close
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty("Pose Estimation with Angles", cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Pose estimation complete. Video saved at: {output_video_path}")
    print(f"Keypoints TSV saved at: {keypoints_tsv_path}")
    print(f"Angles TSV saved at: {angles_tsv_path}")

    plot_angle_changes_over_time(angles_tsv_path, angles_plt_path)

    # Months X-axis
    months = np.linspace(0, 6, 100)
    # Curves
    p90 = create_curve([0.2, 2.5, 3.5, 5.5, 6], [4, 12, 14, 26, 26.5], months)
    p75 = create_curve([0.5, 1.3, 2.5, 3.5, 5.5, 6], [4, 7, 11, 13, 24, 25], months)
    p50 = create_curve([0.6, 3.2, 6], [4, 12, 22], months)
    p25 = create_curve([0.6, 3.5, 4.5, 5.5, 6], [4, 12, 13, 20, 21.5], months)
    p10 = create_curve([0.6, 1.5, 3.4, 4.5, 5.5, 6], [4, 5, 11, 12, 19, 20.25], months)
    p5 = create_curve([0.6, 1.5, 3.4, 4.5, 5.5, 6], [4, 5, 9, 10, 19, 20.25], months)

    # Baby's biological age
    baby_age_months = round(
        (datetime.datetime.today() - datetime.datetime.strptime(birthday.get(), "%d/%m/%y")).days / 30.44, 2)
    baby_score = score_sum(keypoints_tsv_path, angles_tsv_path)  # first 8 basic steps until 2 months

    report_data = generate_aims_report([PRONE_SCORE, SUPINE_SCORE, SITTING_SCORE, STANDING_SCORE])
    curves = {"p90": p90, "p75": p75, "p50": p50, "p25": p25, "p10": p10, "p5": p5}
    plot_aims_report(months, curves, baby_age_months, baby_score, report_data, expert_plt_path)
    plot_aims_percentile_bar(baby_age_months, baby_score, curves, parent_plt_path)


# ---------------- GUI WELCOME PAGE ----------------

video_path = None
video_loaded = False

def check_age():
    try:
        birth_date = datetime.datetime.strptime(birthday.get(), "%d/%m/%y")
        age_days = (datetime.datetime.today() - birth_date).days
        age_months = round(age_days / 30.44, 2)

        if age_days < 0:
            age_status_label.config(
                text=f"Biological Age: {age_months} months\n\nAge is positive value", fg="red")
            set_generate_button(enabled=False)

        elif age_days < 60:
            age_status_label.config(
                text=f"Biological Age: {age_months} months\n\nToo young (< 2 months)", fg="red")
            set_generate_button(enabled=False)

        elif age_days > 180:
            age_status_label.config(
                text=f"Biological Age: {age_months} months\n\nToo old (> 6 months)", fg="red")
            set_generate_button(enabled=False)

        else:
            age_status_label.config(
                text=f"Biological Age: {age_months} months\n\nValid age", fg="green")
            set_generate_button(enabled=True)

    except Exception:
        age_status_label.config(text="Invalid date.", fg="red")
        set_generate_button(enabled=False)

def set_generate_button(enabled):
    if enabled:
        generate_button.config(state=tk.NORMAL, bg="#28a745", fg="white", cursor="hand2")
    else:
        generate_button.config(state=tk.DISABLED, bg="lightgray", fg="gray", cursor="arrow")

def open_video(pose):
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if file_path:
        pose_videos[pose]["path"] = file_path
        pose_videos[pose]["loaded"] = True
        pose_videos[pose]["success_lbl"].config(text="Video loaded successfully!")
        pose_videos[pose]["success_lbl"].grid()
        pose_videos[pose]["show_btn"].grid()

def on_drop(event, pose):
    path = event.data.strip('{}')
    if os.path.exists(path):
        pose_videos[pose]["path"] = path
        pose_videos[pose]["loaded"] = True
        pose_videos[pose]["success_lbl"].config(text="Video loaded successfully!")
        pose_videos[pose]["success_lbl"].grid()
        pose_videos[pose]["show_btn"].grid()

def show_video(pose):
    path = pose_videos[pose]["path"]
    if path:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Unable to open {pose} video.")
            return
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(f"{pose} Video", frame)
            if cv2.waitKey(25) & 0xFF == ord('q') or cv2.getWindowProperty(f"{pose} Video", cv2.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
        cv2.destroyAllWindows()

# Build the GUI
root = TkinterDnD.Tk()
root.title("Infant Motor Development Analyzer")

# Set fixed window size
window_width = 600
window_height = 700
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate x and y coordinates for the window to be centered
x_cordinate = int((screen_width / 2) - (window_width / 2))
y_cordinate = int((screen_height / 2) - (window_height / 2))

root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
root.resizable(False, False)


root.configure(bg="#f0f7ff")  # Soft gradient-like background

prone_img = tk.PhotoImage(file="assets\infant_prone.png").subsample(2, 2)
supine_img = tk.PhotoImage(file="assets\infant_supine.png").subsample(2, 2)
sitting_img = tk.PhotoImage(file="assets\infant_sitting.png").subsample(2, 2)
birthday = tk.StringVar()

#for now
default_date = datetime.datetime.today() - datetime.timedelta(days=100)
birthday.set(default_date.strftime("%d/%m/%y"))

# Title
tk.Label(root, text="Infant Motor Development Analyzer",
         font=("Helvetica", 20, "bold"), bg="#f0f7ff", fg="#003366").pack(pady=(15, 10))

# Birthday label + calendar
tk.Label(root, text="Choose Infant Birthday:", font=("Segoe UI", 15), bg="#f0f7ff", fg="#002244").pack(pady=(10, 5))

calendar = DateEntry(root,
    textvariable=birthday,
    width=14,
    background="#0066cc",      # darker blue
    foreground="white",
    borderwidth=0,             # flat modern look
    font=("Segoe UI", 11),     # modern font
    date_pattern='dd/mm/yy',
    selectbackground="#3399ff",
    selectforeground="white"
)

calendar.pack(pady=5)
calendar.bind("<<DateEntrySelected>>", lambda e: check_age())

age_status_label = tk.Label(root, text="", font=("Segoe UI", 10), bg="#f0f7ff")
age_status_label.pack()


# Video Section
tk.Label(root, text="Load Infant Videos:", font=("Segoe UI", 15), bg="#f0f7ff", fg="#002244").pack(pady=(25, 5))

def styled_button(text, command):
    return tk.Button(root, text=text, command=command, font=("Segoe UI", 10, "bold"),
                     bg="#ffffff", fg="#003366", activebackground="#d0e8ff",
                     relief="raised", bd=2, padx=10, pady=5)

# Section for Prone, Supine, Sitting video input
drop_frame = tk.Frame(root, bg="#f0f7ff")
drop_frame.pack(pady=20)


def create_video_input(title, drop_command, open_command, show_command, image_path, column_index):
    img = tk.PhotoImage(file=image_path).subsample(2, 2) if os.path.exists(image_path) else None

    # Title
    tk.Label(drop_frame, text=title, font=("Segoe UI", 11, "bold"), bg="#f0f7ff", fg="#003366")\
        .grid(row=0, column=column_index, pady=(0, 5))

    # Image
    if img:
        img_label = tk.Label(drop_frame, image=img, bg="#f0f7ff")
        img_label.image = img
        img_label.grid(row=1, column=column_index, pady=(0, 5))

    # Open button
    tk.Button(drop_frame, text=f"Open {title} Video",
              command=lambda: open_video(title),
              font=("Segoe UI", 9), bg="white", fg="#003366")\
        .grid(row=2, column=column_index, pady=(0, 5))

    # Drop zone
    drop_area = tk.Label(drop_frame, text="Drop Video Here", width=18, height=5,
                         bg="#d0e8ff", relief="ridge", bd=2, font=("Segoe UI", 9, "bold"))
    drop_area.grid(row=3, column=column_index, pady=(0, 5), padx=15)
    drop_area.drop_target_register(DND_FILES)
    drop_area.dnd_bind('<<Drop>>', lambda e: on_drop(e, title))

    # Show button (hidden by default)
    show_btn = tk.Button(drop_frame, text=f"Show {title} Video",
                         command=lambda: show_video(title),
                         font=("Segoe UI", 9), bg="#e6f2ff", fg="black",
                         relief="groove", bd=1)
    show_btn.grid(row=4, column=column_index, pady=(0, 2))
    show_btn.grid_remove()

    # Success label (hidden by default)
    success_lbl = tk.Label(drop_frame, text="", font=("Segoe UI", 9), bg="#f0f7ff", fg="green")
    success_lbl.grid(row=5, column=column_index)
    success_lbl.grid_remove()

    # Save references
    pose_videos[title]["show_btn"] = show_btn
    pose_videos[title]["success_lbl"] = success_lbl


pose_videos = {
    "Prone": {"path": None, "loaded": False},
    "Supine": {"path": None, "loaded": False},
    "Sitting": {"path": None, "loaded": False},
}

# Replace these paths with your actual images
create_video_input("Prone", on_drop, open_video, show_video, "assets\infant_prone.png", 0)
create_video_input("Supine", on_drop, open_video, show_video, "assets\infant_supine.png", 1)
create_video_input("Sitting", lambda e: print("Sitting drag"), lambda: print("Sitting open"), lambda: print("Sitting show"), "assets\infant_sitting.png", 2)

success_label = tk.Label(root, text="", font=("Segoe UI", 11), bg='#f0f7ff', fg="green")
success_label.pack()

show_button = tk.Button(root, text="Show Video", command=show_video,
                        font=("Segoe UI", 10), bg="#e6f2ff", fg="black", relief="groove", bd=1)
show_button.pack_forget()

generate_button = tk.Button(root, text="Generate", command=lambda: run_main_process("Prone"),
                            font=("Segoe UI", 11, "bold"), bg="lightgray", fg="gray",
                            activebackground="#218838", relief="flat", bd=0, padx=10, pady=6,
                            state=tk.DISABLED, cursor="arrow")
generate_button.pack(side="bottom", pady=15)


def run_main_process(pose):
    video_path = pose_videos[pose]["path"]
    if not video_path:
        messagebox.showerror("Error", f"No video loaded for {pose}.")
        return

    #trained_model_path = "./runs/pose_train/weights/best.pt"
    prone_trained_model_path = f'final_models/best_prone_1.6.25.pt'
    supine_trained_model_path = f"final_models/best_supine_16.10.25.pt"
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    time_stamp = datetime.datetime.now().strftime("%H:%M")
    asses_dir = f"videos_output"
    pe_trained_video_path = f"{asses_dir}/PE_trained_{video_name}.mp4"
    keypoints_tsv_path = f"{asses_dir}/Keypoints_trained_{video_name}.tsv"
    angles_tsv_path = f"{asses_dir}/Angles_trained_{video_name}.tsv"
    angles_plt_path = f"{asses_dir}/Angles_Graph_{video_name}.jpg"
    expert_plt_path = f"{asses_dir}/Expert_Side_{video_name}.jpg"
    parent_plt_path = f"{asses_dir}/Parent_Side_{video_name}.jpg"

    try:
        save_first_frame_keypoints(prone_trained_model_path, video_path)
        test_model_with_angles(prone_trained_model_path, video_path, pe_trained_video_path, keypoints_tsv_path, angles_tsv_path, angles_plt_path, expert_plt_path, parent_plt_path)
        knn_impute_keypoints_tsv(keypoints_tsv_path)
        messagebox.showinfo("Success", "Processing complete!")
    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong: {e}")

root.mainloop()
