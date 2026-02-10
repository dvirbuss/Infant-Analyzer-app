import tempfile
from pathlib import Path

def save_uploaded_video(uploaded_file, subdir_name="uploads") -> Path:
    base = Path(tempfile.gettempdir()) / "infant_analyzer" / subdir_name
    base.mkdir(parents=True, exist_ok=True)

    out_path = base / uploaded_file.name

    # ✅ SAFE: does not depend on file pointer state
    data = uploaded_file.getvalue()
    out_path.write_bytes(data)

    if out_path.stat().st_size == 0:
        raise ValueError(f"Saved video looks empty: {out_path} (0 bytes)")

    return out_path
