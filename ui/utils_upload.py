import tempfile
from pathlib import Path


def save_video_bytes(video_bytes: bytes, filename: str, subdir_name: str = "uploads") -> Path:
    base = Path(tempfile.gettempdir()) / "infant_analyzer" / subdir_name
    base.mkdir(parents=True, exist_ok=True)

    out_path = base / filename
    out_path.write_bytes(video_bytes)

    if out_path.stat().st_size == 0:
        raise ValueError(f"Saved file is empty: {out_path}")

    return out_path
