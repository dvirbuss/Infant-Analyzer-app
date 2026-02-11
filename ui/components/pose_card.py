# ui2/components/pose_card.py
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Dict, Any

import streamlit as st
from PIL import Image


# ---------- Types ----------
SaveFn = Callable[[bytes, str, str], Path]
# save_fn(video_bytes, filename, subdir_name) -> Path to saved file


@dataclass(frozen=True)
class PoseSpec:
    key: str                 # "prone" | "supine" | "sitting"
    title: str               # "Prone" | ...
    icon_path: Path          # Path to png
    enabled: bool = True
    note: Optional[str] = None


# ---------- Helpers ----------
def load_icon_box(path: Path, box_size=(200, 200)) -> Image.Image:
    """Center icon in a transparent fixed canvas (so all icons look consistent)."""
    img = Image.open(path).convert("RGBA")
    canvas = Image.new("RGBA", box_size, (255, 255, 255, 0))
    img.thumbnail(box_size, Image.Resampling.LANCZOS)
    x = (box_size[0] - img.width) // 2
    y = (box_size[1] - img.height) // 2
    canvas.paste(img, (x, y), img)
    return canvas


def _video_preview_bytes_small(video_bytes: bytes, height_px: int = 220) -> None:
    """Fixed-height preview that doesn't change layout."""
    if not video_bytes:
        return
    b64 = base64.b64encode(video_bytes).decode()

    st.markdown(
        f"""
        <div class="video-box" style="height:{height_px}px;">
          <video controls>
            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
          </video>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Main component ----------
def render_pose_card(
    spec: PoseSpec,
    save_fn: SaveFn,
    subdir_name: str = "uploads",
    preview_height_px: int = 210,
) -> Dict[str, Any]:
    """
    Renders one pose upload card and returns state:
      {
        "uploaded_file": UploadedFile | None,
        "saved_path": Path | None,
        "saved_bytes_len": int,
        "confirmed": bool,
      }
    """
    # Session keys
    k_file = f"{spec.key}__file"
    k_saved = f"{spec.key}__saved_path"
    k_confirmed = f"{spec.key}__confirmed"
    k_last_name = f"{spec.key}__last_name"

    if k_saved not in st.session_state:
        st.session_state[k_saved] = None
    if k_confirmed not in st.session_state:
        st.session_state[k_confirmed] = False
    if k_last_name not in st.session_state:
        st.session_state[k_last_name] = None

    st.markdown('<div class="pose-card">', unsafe_allow_html=True)

    # Title + note
    st.markdown(f'<div class="pose-title">{spec.title} Video</div>', unsafe_allow_html=True)
    if spec.note:
        st.markdown(f'<div class="pose-note">{spec.note}</div>', unsafe_allow_html=True)

    # Centered icon block (fixed height)
    st.markdown('<div class="pose-icon-wrap">', unsafe_allow_html=True)
    if spec.icon_path and Path(spec.icon_path).exists():
        st.image(load_icon_box(Path(spec.icon_path)), use_column_width=False)
    else:
        st.markdown("<div style='color:#888;'>Icon missing</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Uploader (kept always visible)
    uploaded = st.file_uploader(
        label=f"{spec.title} Video",
        type=["mp4", "mpeg4", "mov"],
        key=k_file,
        disabled=not spec.enabled,
        label_visibility="collapsed",
    )

    # Detect new file selection -> reset previous confirmation/save
    if uploaded is not None and uploaded.name != st.session_state[k_last_name]:
        st.session_state[k_last_name] = uploaded.name
        st.session_state[k_saved] = None
        st.session_state[k_confirmed] = False

    # Preview + confirm/save
    if uploaded is not None:
        # IMPORTANT: read bytes once and reuse (prevents 0-byte saves)
        video_bytes = uploaded.getvalue()
        bytes_len = len(video_bytes)

        st.caption("Preview:")
        _video_preview_bytes_small(video_bytes, height_px=220)

        # Buttons row
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button(f"✅ Confirm & Save {spec.title}", key=f"{spec.key}__save_btn"):
                if bytes_len <= 0:
                    st.error("Upload looks empty (0 bytes). Try uploading again.")
                else:
                    saved_path = save_fn(video_bytes, uploaded.name, subdir_name)
                    st.session_state[k_saved] = saved_path
                    st.session_state[k_confirmed] = True

        with b2:
            if st.button("🧹 Clear", key=f"{spec.key}__clear_btn"):
                st.session_state[k_saved] = None
                st.session_state[k_confirmed] = False
                st.session_state[k_last_name] = None
                st.rerun()

    else:
        bytes_len = 0

    # Saved indicator
    saved_path = st.session_state[k_saved]
    confirmed = st.session_state[k_confirmed]
    if confirmed and saved_path:
        st.success(f"Saved: {saved_path.name}")

    st.markdown("</div>", unsafe_allow_html=True)

    return {
        "uploaded_file": uploaded,
        "saved_path": saved_path,
        "saved_bytes_len": bytes_len,
        "confirmed": confirmed,
    }
