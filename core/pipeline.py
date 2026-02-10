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


def run(pose: str, video_path: str, birthdate: dt.date, out_dir: Path) -> dict:
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
    save_first_frame_keypoints(Path(model_path), video_path)

    # 2) inference -> artifacts
    artifacts = infer_video_with_angles(model, video_path, out_dir)

    # 3) impute keypoints in-place
    knn_impute_keypoints_tsv(artifacts["keypoints_tsv"])

    # 4) scoring
    scores = score_all(artifacts["keypoints_tsv"], artifacts["angles_tsv"])
    baby_age_months = round(((dt.date.today() - birthdate).days) / 30.44, 2)

    # support both dataclass(.total()) and plain dict/tuple
    baby_score = scores.total() if hasattr(scores, "total") else int(
        sum(map(int, scores["prone"] + scores["supine"] + scores["sitting"] + scores["standing"]))
    )

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
    }
