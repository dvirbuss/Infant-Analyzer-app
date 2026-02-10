import tempfile
import datetime
from pathlib import Path

import streamlit as st
import config

from ui.styles import base_css, generate_button_css
from ui.components import load_icon_box
from core.pipeline import run
from ui.utils_upload import save_uploaded_video

def render_app():
    st.set_page_config(page_title="Infant Motor Development Analyzer", layout="wide")
    st.markdown(base_css(), unsafe_allow_html=True)

    st.markdown("<div class='app-title'>Infant Motor Development Analyzer</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-subtitle'>Choose infant birthday and load pose videos</div>", unsafe_allow_html=True)

    # --- Birthday ---
    left, center, right = st.columns([5, 2, 5])
    with center:
        st.markdown("<div style='text-align:center;'>Choose Infant Birthday:</div>", unsafe_allow_html=True)
        birth_date = st.date_input(
            "Infant birthday",
            value=datetime.date.today() - datetime.timedelta(days=60),
            format="DD/MM/YYYY",
            label_visibility="collapsed"
        )

    # --- Upload cards ---
    st.markdown("<h4 style='text-align:center;'>Load Infant Videos:</h4>", unsafe_allow_html=True)
    cols = st.columns(3, gap="large")
    uploaded = {}
    if "saved_prone_path" not in st.session_state:
        st.session_state.saved_prone_path = None

    with cols[0]:
        st.markdown("<div class='pose-card'>", unsafe_allow_html=True)
        st.image(load_icon_box(config.PRONE_ICON))

        uploaded["prone"] = st.file_uploader("Prone Video", type=["mp4"], key="prone_uploader")

        if uploaded["prone"] is not None and uploaded["prone"].size == 0:
            st.error("Upload returned 0 bytes. Refresh the page (F5) and re-upload.")
            st.stop()

        # ✅ PREVIEW GOES EXACTLY HERE (inside the prone card)
        if uploaded["prone"] is not None:
            st.markdown("**Preview:**")
            st.video(uploaded["prone"].getvalue())

            # Optional: confirm-save button right under the preview
            if st.button("✅ Confirm & Save Prone Video", key="save_prone_btn"):
                saved_path = save_uploaded_video(uploaded["prone"], subdir_name="uploads")
                st.session_state.saved_prone_path = str(saved_path)
                st.success(f"Saved: {saved_path} ({Path(saved_path).stat().st_size / 1_048_576:.2f} MB)")

        st.markdown("</div>", unsafe_allow_html=True)

    with cols[1]:
        st.markdown("<div class='pose-card'>", unsafe_allow_html=True)
        st.image(load_icon_box(config.SUPINE_ICON))
        uploaded["Supine"] = st.file_uploader("Supine Video (future)", type=["mp4"])
        st.markdown("</div>", unsafe_allow_html=True)

    with cols[2]:
        st.markdown("<div class='pose-card'>", unsafe_allow_html=True)
        st.image(load_icon_box(config.SITTING_ICON))
        uploaded["Sitting"] = st.file_uploader("Sitting Video (future)", type=["mp4"])
        st.markdown("</div>", unsafe_allow_html=True)

    # For now: only Prone
    pose = "prone"
    enabled = uploaded[pose] is not None
    st.markdown(generate_button_css(enabled), unsafe_allow_html=True)

    btn_cols = st.columns([3, 2, 3])
    with btn_cols[1]:
        run_button = st.button("Generate", disabled=not enabled)

    # --- Run pipeline ---
    if run_button and uploaded[pose] is not None:
        with st.spinner("Processing..."):
            tmp_dir = Path(tempfile.mkdtemp())
            in_path = tmp_dir / uploaded[pose].name
            in_path.write_bytes(uploaded[pose].read())
            out_dir = tmp_dir / "outputs"

            result = run(
                pose=pose,
                video_path=str(in_path),
                birthdate=birth_date,
                out_dir=out_dir
            )

        st.success("Done!")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Age (months)", result["age_months"])
            st.metric("AIMS Score", result["aims_score"])
            st.video(result["artifacts"]["video_out"])

        with c2:
            if "angles_plot" in result["reports"]:
                st.image(result["reports"]["angles_plot"], caption="Angles over time", use_container_width=True)
            st.image(result["reports"]["expert_plot"], caption="Expert report", use_container_width=True)
            st.image(result["reports"]["parent_plot"], caption="Parent report", use_container_width=True)
