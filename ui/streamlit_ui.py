import datetime
import streamlit as st
from pathlib import Path

# Internal imports
import config
from ui.styles import base_css, generate_button_css
from ui.components.pose_card import render_pose_card, PoseSpec
from ui.utils_upload import save_video_bytes
from core.pipeline import run as run_pipeline


def render_app():
    # --- Setup ---
    st.set_page_config(page_title="Infant Motor Development Analyzer", layout="wide")
    st.markdown(base_css(), unsafe_allow_html=True)

    st.markdown("<div class='app-title'>Infant Motor Development Analyzer</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-subtitle'>Choose infant birthday and load pose videos</div>", unsafe_allow_html=True)

    # --- 1. Birthday (Centered) ---
    _, center_date, _ = st.columns([5, 2, 5])
    with center_date:
        st.markdown("<h4 style='text-align:center;'>Choose Infant Birthday:</h4>", unsafe_allow_html=True)
        birth_date = st.date_input(
            "Infant birthday",
            value=datetime.date.today() - datetime.timedelta(days=60),
            format="DD/MM/YYYY",
            label_visibility="collapsed"
        )

    # --- 2. Render Cards ---
    poses = [
        PoseSpec(key="prone", title="Prone", icon_path=config.PRONE_ICON),
        PoseSpec(key="supine", title="Supine", icon_path=config.SUPINE_ICON),
        PoseSpec(key="sitting", title="Sitting", icon_path=config.SITTING_ICON),
    ]

    st.markdown("<h4 style='text-align:center;'>Load Infant Videos:</h4>", unsafe_allow_html=True)
    cols = st.columns(3, gap="large")

    for i, spec in enumerate(poses):
        with cols[i]:
            # Safety check: Prevent crash if icon is missing
            if not spec.icon_path.exists():
                st.warning(f"File not found: {spec.icon_path.name}")

            render_pose_card(
                spec=spec,
                save_fn=save_video_bytes,
                subdir_name=f"infant_{birth_date.strftime('%Y%m%d')}"
            )

    # --- 3. Generate Button (Centered & Green) ---
    # Check session state ONLY after all cards have been rendered
    confirmed_keys = [k for k in ["prone", "supine", "sitting"] if st.session_state.get(f"{k}__confirmed")]
    any_confirmed = len(confirmed_keys) > 0

    # Centered button column
    _, btn_col, _ = st.columns([5, 2, 5])

    with btn_col:
        # Use your existing CSS function
        st.markdown(generate_button_css(any_confirmed), unsafe_allow_html=True)

        if st.button("GENERATE", disabled=not any_confirmed):
            progress_text = st.empty()

            pose_map = {
                "prone": "Prone",
                "supine": "Supine",
                "sitting": "Sitting",
            }

            results = {}

            with st.expander("Debug session_state"):
                for k in ["prone", "supine", "sitting"]:
                    st.write(k, "confirmed =", st.session_state.get(f"{k}__confirmed"))
                    st.write(k, "path =", st.session_state.get(f"{k}__path"))

            for pose_key in confirmed_keys:
                pose_title = pose_map[pose_key]

                video_path = st.session_state.get(f"{pose_key}__path", None)
                if video_path is not None:
                    video_path = str(video_path)

                if not video_path:
                    st.error(f"{pose_title}: Missing saved video path in session_state ({pose_key}__path).")
                    continue

                progress_text.info(f"Running analysis for {pose_title}...")

                try:
                    result = run_pipeline(
                        pose=pose_title,
                        video_path=video_path,
                        birthdate=birth_date,
                        runner="streamlit",
                    )
                    results[pose_key] = result
                    st.success(f"{pose_title}: Done ✅ (AIMS={result['aims_score']})")

                except Exception as e:
                    st.error(f"Error in {pose_title}: {e}")

            progress_text.success("All analyses complete!")
            st.balloons()
    # --- 4. DEBUG SECTION (See where it's looking) ---
    with st.expander("🛠️ Debug Path Info"):
        st.write(f"**BASE_DIR:** `{config.BASE_DIR}`")
        st.write(f"**ASSETS_DIR:** `{config.ASSETS_DIR}`")
        for p in poses:
            exists = "✅ Found" if p.icon_path.exists() else "❌ NOT FOUND"
            st.write(f"**{p.title} Icon Path:** `{p.icon_path}` | {exists}")