def base_css() -> str:
    return """
    <style>
    .app-title { text-align:center; font-size:40px; font-weight:700; margin-top:1rem; }
    .app-subtitle { text-align:center; font-size:22px; color:#444; margin-bottom:2rem; }
    .pose-card { background:#f7fbff; border-radius:12px; padding:1.2rem 1rem 1.5rem;
                 box-shadow:0 2px 6px rgba(0,0,0,0.05); border:1px solid #d0e2ff; text-align:center; }
    .pose-card img { height:150px !important; width:auto !important; object-fit:contain; }
    </style>
    """

def generate_button_css(enabled: bool) -> str:
    if enabled:
        return """
        <style>
        div.stButton > button:first-child {
            background-color:#4CAF50 !important;
            color:white !important;
            border-radius:20px !important;
            width:260px !important;
            height:70px !important;
            font-size:32px !important;
            font-weight:700 !important;
        }
        </style>
        """
    return """
    <style>
    div.stButton > button:first-child {
        background-color:#d9d9d9 !important;
        color:#666 !important;
        border-radius:12px !important;
        width:260px !important;
        height:70px !important;
        font-size:32px !important;
        font-weight:700 !important;
        cursor:not-allowed !important;
    }
    </style>
    """
