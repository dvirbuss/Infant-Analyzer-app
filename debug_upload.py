import tempfile
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Upload Debug")

st.title("Upload Debug")

f = st.file_uploader("Upload MP4", type=["mp4"])

if f is None:
    st.info("Waiting for upload...")
    st.stop()

st.write("name:", f.name)
st.write("size (streamlit):", f.size)

data = f.getvalue()
st.write("len(getvalue):", len(data))

# save to temp to verify
out_dir = Path(tempfile.gettempdir()) / "infant_analyzer_debug"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f.name
out_path.write_bytes(data)

st.write("saved_to:", str(out_path))
st.write("saved_size:", out_path.stat().st_size)
st.success("Done")
