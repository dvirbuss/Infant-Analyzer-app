import tempfile
import datetime
from pathlib import Path
from PIL import Image

import streamlit as st
import numpy as np
import pandas as pd
import cv2

import config  # your config file with ICON paths


# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Infant Motor Development Analyzer",
    layout="wide"
)

# ---------------------- CSS ------------------------------
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
    width: 50px !important;
}

.compact-date input {
    width: 50px !important;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ---------------------- TITLE ----------------------------
st.markdown("<div class='app-title'>Infant Motor Development Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Choose infant birthday and load pose videos</div>", unsafe_allow_html=True)


# ---------------------- BIRTH DATE ------------------------
left, center, right = st.columns([5, 2, 5])
with center:
    st.markdown("<div style='text-align:center;'>Choose Infant Birthday:</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='compact-date'>", unsafe_allow_html=True)

        birth_date = st.date_input(
            "",
            value=datetime.date.today() - datetime.timedelta(days=60),
            format="DD/MM/YYYY"
        )

        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- IMAGE BOX LOADER ------------------
def load_icon_box(path, box_size=(200, 200)):
    img = Image.open(path).convert("RGBA")
    canvas = Image.new("RGBA", box_size, (255, 255, 255, 0))
    img.thumbnail(box_size, Image.Resampling.LANCZOS)

    x = (box_size[0] - img.width) // 2
    y = (box_size[1] - img.height) // 2
    canvas.paste(img, (x, y), img)

    return canvas


# ---------------------- VIDEO CARDS -----------------------
st.markdown("<h4 style='text-align:center;'>Load Infant Videos:</h4>", unsafe_allow_html=True)
cols = st.columns(3, gap="large")

uploaded = {}

with cols[0]:
    st.markdown("<div class='pose-card'>", unsafe_allow_html=True)
    st.image(load_icon_box(config.PRONE_ICON))
    uploaded["prone"] = st.file_uploader("Prone Video", type=["mp4"])
    st.markdown("</div>", unsafe_allow_html=True)

with cols[1]:
    st.markdown("<div class='pose-card'>", unsafe_allow_html=True)
    st.image(load_icon_box(config.SUPINE_ICON))
    uploaded["supine"] = st.file_uploader("Supine Video (future)", type=["mp4"])
    st.markdown("</div>", unsafe_allow_html=True)

with cols[2]:
    st.markdown("<div class='pose-card'>", unsafe_allow_html=True)
    st.image(load_icon_box(config.SITTING_ICON))
    uploaded["sitting"] = st.file_uploader("Sitting Video (future)", type=["mp4"])
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- GENERATE BUTTON -------------------
generate_disabled = uploaded["prone"] is None
st.markdown("<div class='centered'>", unsafe_allow_html=True)
run_button = st.button("Generate", disabled=generate_disabled)
st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- RUN ANALYSIS ----------------------
if run_button and uploaded["prone"] is not None:

    prone_file = uploaded["prone"]

    with st.spinner("Processing prone video... this may take a bit"):
        tmp_dir = Path(tempfile.mkdtemp())
        prone_path = tmp_dir / prone_file.name
        with open(prone_path, "wb") as f:
            f.write(prone_file.read())

        # 📌 PLACE YOUR ANALYSIS FUNCTION HERE
        # result = run_prone(str(prone_path), birth_date)

    st.success("Processing complete! (analysis logic not activated yet)")
