import tempfile
import datetime
from pathlib import Path

import streamlit as st

from PIL import Image

# from core.pipeline import run_prone
import config  # 👈 import your config module

st.set_page_config(
    page_title="Infant Motor Development Analyzer",
    layout="wide"
)

# ---------- CSS omitted here for brevity ----------
st.markdown("""
<style>

.app-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    margin-top: 1rem;
    margin-bottom: 0.25rem;
}

.app-subtitle {
    text-align: center;
    font-size: 22px;
    color: #444444;
    margin-bottom: 2rem;
}

.pose-card {
    background-color: #f7fbff;
    border-radius: 12px;
    padding: 1.2rem 1rem 1.5rem 1rem;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    border: 1px solid #d0e2ff;
    text-align: center;
}

/* FIX ICON SIZE */
.pose-card img {
    height: 150px !important;
    width: auto !important;
    object-fit: contain;
}

.centered {
    display: flex;
    justify-content: center;
    align-items: center;
}

.compact-date .stDateInput {
    width: 50px !important;   /* adjust width */
}

.compact-date input {
    width: 50px !important;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)



# ---------- TITLE ----------
st.markdown("<div class='app-title'>Infant Motor Development Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Choose infant birthday and load pose videos</div>", unsafe_allow_html=True)

# ---------- BIRTHDAY ----------
# Centered container for the birthday field
st.markdown("<br>", unsafe_allow_html=True)

# Three columns: wide, narrow, wide
left, center, right = st.columns([5, 2, 5])

with center:
    # Centered label
    st.markdown("<div style='text-align:center;'>Choose Infant Birthday:</div>", unsafe_allow_html=True)

    # Wrap the date input in a compact class
    with st.container():
        st.markdown("<div class='compact-date'>", unsafe_allow_html=True)

        birth_date = st.date_input(
            "",
            value=datetime.date.today() - datetime.timedelta(days=60),
            format="DD/MM/YYYY"
        )

        st.markdown("</div>", unsafe_allow_html=True)


# ---------- VIDEO CARDS ----------

st.markdown("<h4 style='text-align:center;'>Load Infant Videos:</h4>", unsafe_allow_html=True)
cols = st.columns(3, gap="large")

uploaded = {}


def load_icon_box(path, box_size=(200, 200)):
    """
    Loads an image and centers it inside a fixed-size box.
    Ensures ALL icons have identical width and height visually.
    """
    img = Image.open(path).convert("RGBA")
    canvas = Image.new("RGBA", box_size, (255, 255, 255, 0))  # transparent background

    # resize while keeping aspect ratio
    img.thumbnail(box_size, Image.Resampling.LANCZOS)

    # center the icon
    x = (box_size[0] - img.width) // 2
    y = (box_size[1] - img.height) // 2
    canvas.paste(img, (x, y), img)

    return canvas

with cols[0]:
    st.markdown("<div class='pose-card'>", unsafe_allow_html=True)
    st.image(load_icon_box(config.PRONE_ICON), use_container_width=False)    # 👈 from config
    st.markdown("<div class='pose-title'>Prone</div>", unsafe_allow_html=True)
    uploaded["prone"] = st.file_uploader(
        "Drop Video Here",
        type=["mp4"],
        key="prone_uploader",
        label_visibility="collapsed"
    )
    st.caption("Open Prone Video / Drop MP4")
    st.markdown("</div>", unsafe_allow_html=True)

with cols[1]:
    st.markdown("<div class='pose-card'>", unsafe_allow_html=True)
    st.image(load_icon_box(config.SUPINE_ICON), use_container_width=False)     # 👈 from config
    st.markdown("<div class='pose-title'>Supine</div>", unsafe_allow_html=True)
    uploaded["supine"] = st.file_uploader(
        "Drop Video Here",
        type=["mp4"],
        key="supine_uploader",
        label_visibility="collapsed"
    )
    st.caption("Open Supine Video / Drop MP4 (future)")
    st.markdown("</div>", unsafe_allow_html=True)

with cols[2]:
    st.markdown("<div class='pose-card'>", unsafe_allow_html=True)
    st.image(load_icon_box(config.SITTING_ICON), use_container_width=False)   # 👈 from config
    st.markdown("<div class='pose-title'>Sitting</div>", unsafe_allow_html=True)
    uploaded["sitting"] = st.file_uploader(
        "Drop Video Here",
        type=["mp4"],
        key="sitting_uploader",
        label_visibility="collapsed"
    )
    st.caption("Open Sitting Video / Drop MP4 (future)")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- GENERATE BUTTON ----------
generate_disabled = uploaded["prone"] is None
st.markdown("<div class='centered'>", unsafe_allow_html=True)
run_button = st.button("Generate", disabled=generate_disabled)
st.markdown("</div>", unsafe_allow_html=True)

# ---------- RUN ANALYSIS ----------
if run_button and uploaded["prone"] is not None:
    prone_file = uploaded["prone"]
    with st.spinner("Processing prone video... this may take a bit"):
        tmp_dir = Path(tempfile.mkdtemp())
        prone_path = tmp_dir / prone_file.name
        with open(prone_path, "wb") as f:
            f.write(prone_file.read())

        # result = run_prone(str(prone_path), birth_date)

    # st.success("Analysis complete ✅")
    # st.write(f"**Biological age:** {result['age_months']} months")
    # st.write(f"**AIMS total score:** {result['total_score']}")
    #
    # st.subheader("Expert report")
    # st.image(str(result["expert_plot"]))
    #
    # st.subheader("Parent-friendly percentile bar")
    # st.image(str(result["parent_plot"]))
    #
    # st.subheader("Pose-estimation video")
    # st.video(str(result["video_output"]))


