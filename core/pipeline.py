# core/pipeline.py
from __future__ import annotations

from pathlib import Path
import datetime as dt
from ultralytics import YOLO

import config
from .video_ops import save_first_frame_keypoints
from .pose_infer import infer_video_with_angles
from .helpers import knn_impute_keypoints_tsv
from .aims_scoring import score_all
from .reporting import build_reports


# --------- lightweight model cache (core-safe, no streamlit import) ---------
_MODEL_CACHE: dict[str, YOLO] = {}


def load_model(model_path: Path) -> YOLO:
    key = str(model_path.resolve())
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = YOLO(str(model_path))
    return _MODEL_CACHE[key]


def run(pose: str, video_path: str, birthdate: dt.date, out_dir: Path | None = None,
    runner: str = "unknown", frame_callback=None, cancel_check=None, progress_callback=None) -> dict:
    if out_dir is None:
        stamp = dt.datetime.now().strftime("%d-%m-%y_%H-%M")
        video_name = Path(video_path).stem
        folder_name = f"{pose.lower()}_{video_name}_{stamp}_{runner}"
        out_dir = Path(config.VIDEOS_OUTPUT_DIR) / folder_name

    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = {
        "Prone": config.PRONE_MODEL_PATH,
        "Supine": config.SUPINE_MODEL_PATH,
        "Sitting": config.PRONE_MODEL_PATH, #todo
    }.get(pose)

    if model_path is None:
        raise ValueError(f"Unsupported pose: {pose}")

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = load_model(Path(model_path))

    # 1) mirror if needed (in-place)
    save_first_frame_keypoints(model, video_path)

    # 2) inference -> artifacts
    artifacts = infer_video_with_angles(model=model, video_path=video_path,
        out_dir=out_dir, frame_callback=frame_callback, cancel_check=cancel_check, progress_callback=progress_callback  # pass through
    )
    frames_processed = artifacts.get("frames_processed", 0)
    baby_age_months = round(((dt.date.today() - birthdate).days) / 30.44, 2)

    if cancel_check is not None and cancel_check():
        return {
            "pose": pose,
            "age_months": baby_age_months,
            "aims_score": None,
            "scores": None,
            "artifacts": artifacts,
            "reports": {},
            "frames_processed": frames_processed,
            "cancelled": True,
        }

    # 3) impute keypoints in-place
    knn_impute_keypoints_tsv(artifacts["keypoints_tsv"])

    # 4) scoring
    scores = score_all(artifacts["keypoints_tsv"], artifacts["angles_tsv"])
    baby_age_months = round(((dt.date.today() - birthdate).days) / 30.44, 2)

    # FIX: Handle baby_score calculation safely
    if hasattr(scores, "total"):
        baby_score = scores.total()
    elif isinstance(scores, dict):
        # Ensure keys exist before summing
        keys = ["prone", "supine", "sitting", "standing"]
        baby_score = int(sum(sum(scores[k]) for k in keys if k in scores))
    else:
        # Fallback if scores is an unexpected type like WindowsPath
        baby_score = 0

    # 5) reports (include angles plot!)
    report_files = build_reports(
        baby_age_months=baby_age_months,
        baby_score=baby_score,
        scores=scores,
        out_dir=out_dir,
        angles_tsv_path=artifacts["angles_tsv"],
    )

    return {
        "pose": pose,
        "age_months": baby_age_months,
        "aims_score": baby_score,
        "scores": scores,
        "artifacts": artifacts,
        "reports": report_files,
        "frames_processed": frames_processed,
        "cancelled": False,
    }
