import os
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from dotenv import load_dotenv

from models.blip_model import get_blip_caption
from models.groq_vision import answer_followup_stream, get_initial_analysis, get_suggested_questions
from utils.image_utils import get_image_info, validate_and_process

load_dotenv()

st.set_page_config(
    page_title="VisualMind AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_api_key(key: str) -> str:
    try:
        val = st.secrets.get(key)
        if val:
            return val
    except Exception:
        pass
    return os.getenv(key, "")


# ── Theme init (must happen before CSS is applied) ────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "light"


def get_css(dark: bool) -> str:
    if dark:
        bg         = "#07070d"
        sb_bg      = "rgba(8,8,18,0.97)"
        sb_border  = "rgba(255,255,255,0.055)"
        surface    = "rgba(255,255,255,0.025)"
        border     = "rgba(255,255,255,0.06)"
        text_p     = "#f1f5f9"
        text_s     = "rgba(248,250,252,0.78)"
        text_m     = "rgba(248,250,252,0.42)"
        text_mm    = "rgba(248,250,252,0.35)"
        text_lbl   = "rgba(248,250,252,0.6)"
        text_slbl  = "rgba(248,250,252,0.65)"
        input_bg   = "rgba(255,255,255,0.038)"
        input_bd   = "rgba(255,255,255,0.09)"
        input_bgf  = "rgba(255,255,255,0.055)"
        dl_bg      = "rgba(255,255,255,0.04)"
        dl_color   = "rgba(248,250,252,0.75)"
        dl_border  = "rgba(255,255,255,0.1)"
        hr_grad    = "transparent, rgba(255,255,255,0.07), transparent"
        hr_sb      = "rgba(255,255,255,0.05)"
        caption_c  = "rgba(248,250,252,0.38)"
        md_p       = "rgba(248,250,252,0.78)"
        exp_sum    = "rgba(248,250,252,0.8)"
        chat_hover = "rgba(255,255,255,0.035)"
        img_shadow = "0 8px 48px rgba(0,0,0,0.45)"
        img_shadow2= "0 14px 64px rgba(0,0,0,0.55)"
    else:
        bg         = "#f0f2f8"
        sb_bg      = "#ffffff"
        sb_border  = "rgba(0,0,0,0.07)"
        surface    = "rgba(0,0,0,0.028)"
        border     = "rgba(0,0,0,0.07)"
        text_p     = "#0f172a"
        text_s     = "rgba(15,23,42,0.72)"
        text_m     = "rgba(15,23,42,0.48)"
        text_mm    = "rgba(15,23,42,0.38)"
        text_lbl   = "rgba(15,23,42,0.6)"
        text_slbl  = "rgba(15,23,42,0.65)"
        input_bg   = "rgba(0,0,0,0.035)"
        input_bd   = "rgba(0,0,0,0.1)"
        input_bgf  = "rgba(124,58,237,0.05)"
        dl_bg      = "rgba(0,0,0,0.04)"
        dl_color   = "rgba(15,23,42,0.65)"
        dl_border  = "rgba(0,0,0,0.1)"
        hr_grad    = "transparent, rgba(0,0,0,0.07), transparent"
        hr_sb      = "rgba(0,0,0,0.06)"
        caption_c  = "rgba(15,23,42,0.4)"
        md_p       = "rgba(15,23,42,0.72)"
        exp_sum    = "rgba(15,23,42,0.8)"
        chat_hover = "rgba(0,0,0,0.035)"
        img_shadow = "0 8px 48px rgba(0,0,0,0.12)"
        img_shadow2= "0 14px 64px rgba(0,0,0,0.2)"

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap');

html, body, .stApp {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif !important;
    background: {bg} !important;
    color: {text_p} !important;
}}
* {{ box-sizing: border-box; }}

#MainMenu {{ display: none !important; }}
footer {{ display: none !important; }}
header {{ display: none !important; }}
.stDeployButton {{ display: none !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}

::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(124,58,237,0.35); border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: rgba(124,58,237,0.6); }}

@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(22px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}
@keyframes glowPulse {{
    0%,100% {{ box-shadow: 0 0 24px rgba(124,58,237,0.08); }}
    50%      {{ box-shadow: 0 0 48px rgba(124,58,237,0.22); }}
}}
@keyframes slideInLeft {{
    from {{ opacity: 0; transform: translateX(-16px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
}}
@keyframes slideInRight {{
    from {{ opacity: 0; transform: translateX(16px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
}}
@keyframes scaleIn {{
    from {{ opacity: 0; transform: scale(0.96); }}
    to   {{ opacity: 1; transform: scale(1); }}
}}

.main .block-container {{
    animation: fadeIn 0.45s ease-out;
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1380px !important;
}}

[data-testid="stSidebar"] {{
    background: {sb_bg} !important;
    border-right: 1px solid {sb_border} !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
}}
[data-testid="stSidebar"] > div:first-child {{ padding-top: 1.5rem; }}
[data-testid="stSidebar"] .stMarkdown p {{
    color: {text_lbl} !important;
    font-size: 0.85rem !important;
    line-height: 1.65 !important;
}}
[data-testid="stSidebar"] label {{
    color: {text_slbl} !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px !important;
}}

h1 {{ font-weight: 800 !important; letter-spacing: -2.5px !important; line-height: 1.05 !important; }}
h2, h3 {{ font-weight: 700 !important; letter-spacing: -0.6px !important; color: {text_p} !important; }}

hr {{
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, {hr_grad}) !important;
    margin: 1.4rem 0 !important;
}}

[data-testid="stFileUploader"] {{ animation: fadeInUp 0.55s ease-out; }}
[data-testid="stFileUploader"] section {{
    background: {surface} !important;
    border: 1.5px dashed rgba(124,58,237,0.38) !important;
    border-radius: 22px !important;
    padding: 2.8rem 2rem !important;
    transition: background 0.25s ease, border-color 0.25s ease, transform 0.25s ease, box-shadow 0.25s ease !important;
    animation: glowPulse 3.5s ease-in-out infinite;
}}
[data-testid="stFileUploader"] section:hover {{
    background: rgba(124,58,237,0.055) !important;
    border-color: rgba(124,58,237,0.65) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 44px rgba(124,58,237,0.14) !important;
}}
[data-testid="stFileUploader"] section > div > span {{ color: {text_s} !important; font-size: 0.95rem !important; }}
[data-testid="stFileUploader"] section > div small {{ color: {text_m} !important; font-size: 0.78rem !important; }}

.stButton > button {{
    background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 13px !important;
    padding: 0.58rem 1.4rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    letter-spacing: -0.1px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 18px rgba(124,58,237,0.28) !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124,58,237,0.45) !important;
    filter: brightness(1.08) !important;
}}
.stButton > button:active {{
    transform: translateY(0) !important;
    box-shadow: 0 3px 12px rgba(124,58,237,0.3) !important;
}}

[data-testid="stPasswordInput"] input,
.stTextInput > div > div > input {{
    background: {input_bg} !important;
    border: 1px solid {input_bd} !important;
    border-radius: 12px !important;
    color: {text_p} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease !important;
}}
[data-testid="stPasswordInput"] input:focus,
.stTextInput > div > div > input:focus {{
    border-color: rgba(124,58,237,0.55) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.13) !important;
    background: {input_bgf} !important;
}}

[data-testid="stChatInput"] {{
    background: {surface} !important;
    border: 1px solid {border} !important;
    border-radius: 18px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    backdrop-filter: blur(12px) !important;
}}
[data-testid="stChatInput"]:focus-within {{
    border-color: rgba(124,58,237,0.48) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.1) !important;
}}
[data-testid="stChatInput"] textarea {{
    background: transparent !important;
    color: {text_p} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
}}

[data-testid="stChatMessage"] {{
    background: {surface} !important;
    border: 1px solid {border} !important;
    border-radius: 18px !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    margin: 0.45rem 0 !important;
    animation: fadeInUp 0.3s ease-out !important;
    transition: border-color 0.2s ease, background 0.2s ease !important;
}}
[data-testid="stChatMessage"]:hover {{
    border-color: rgba(124,58,237,0.18) !important;
    background: {chat_hover} !important;
}}

[data-testid="stAlert"] {{
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
    animation: fadeInUp 0.35s ease-out !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
}}
div[data-testid="stAlert"] {{
    background: rgba(37,99,235,0.07) !important;
    border: 1px solid rgba(37,99,235,0.2) !important;
    color: {text_p} !important;
}}
.element-container div[data-testid="stAlert"][kind="success"] {{
    background: rgba(5,150,105,0.07) !important;
    border: 1px solid rgba(5,150,105,0.2) !important;
}}
[data-testid="stAlert"][kind="warning"] {{
    background: rgba(217,119,6,0.07) !important;
    border: 1px solid rgba(217,119,6,0.22) !important;
    color: #fbbf24 !important;
}}
[data-testid="stAlert"][kind="error"] {{
    background: rgba(220,38,38,0.07) !important;
    border: 1px solid rgba(220,38,38,0.22) !important;
}}

[data-testid="stMetric"] {{
    background: {surface} !important;
    border: 1px solid {border} !important;
    border-radius: 16px !important;
    padding: 1rem 1.1rem !important;
    transition: border-color 0.2s ease, background 0.2s ease, transform 0.2s ease !important;
}}
[data-testid="stMetric"]:hover {{
    border-color: rgba(124,58,237,0.28) !important;
    background: rgba(124,58,237,0.048) !important;
    transform: translateY(-2px) !important;
}}
[data-testid="stMetricValue"] > div {{
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}}
[data-testid="stMetricLabel"] > div {{
    color: {text_m} !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.1px !important;
}}
[data-testid="stMetricDelta"] {{ display: none !important; }}

[data-testid="stImage"] > img {{
    border-radius: 18px !important;
    box-shadow: {img_shadow} !important;
    transition: transform 0.35s ease, box-shadow 0.35s ease !important;
    animation: scaleIn 0.5s ease-out;
}}
[data-testid="stImage"] > img:hover {{
    transform: scale(1.015) !important;
    box-shadow: {img_shadow2} !important;
}}

[data-testid="stCaptionContainer"] p {{
    color: {caption_c} !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1px !important;
}}

.stMarkdown p {{
    color: {md_p} !important;
    line-height: 1.72 !important;
    font-size: 0.93rem !important;
}}

[data-testid="stExpander"] {{
    background: {surface} !important;
    border: 1px solid {border} !important;
    border-radius: 16px !important;
}}
[data-testid="stExpander"] summary {{
    color: {exp_sum} !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}}

[data-testid="stDownloadButton"] button {{
    background: {dl_bg} !important;
    color: {dl_color} !important;
    border: 1px solid {dl_border} !important;
    border-radius: 12px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}}
[data-testid="stDownloadButton"] button:hover {{
    background: rgba(124,58,237,0.1) !important;
    border-color: rgba(124,58,237,0.35) !important;
    color: {text_p} !important;
    transform: translateY(-1px) !important;
}}

[data-testid="stSpinner"] > div {{ border-top-color: #7c3aed !important; }}

[data-testid="column"]:nth-child(1) {{ animation: slideInLeft  0.5s ease-out; }}
[data-testid="column"]:nth-child(2) {{ animation: slideInRight 0.5s ease-out; }}

[data-testid="stSidebar"] hr {{ background: {hr_sb} !important; }}
</style>
"""


st.markdown(get_css(st.session_state.theme == "dark"), unsafe_allow_html=True)

# ── Read API keys silently from environment / Streamlit secrets ───────────────
groq_key = get_api_key("GROQ_API_KEY")
hf_token = get_api_key("HF_TOKEN")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
            <div style="
                width:36px; height:36px; border-radius:10px;
                background: linear-gradient(135deg,#7c3aed,#2563eb);
                display:flex; align-items:center; justify-content:center;
                font-size:18px; flex-shrink:0;
            ">👁️</div>
            <div>
                <div style="
                    font-weight:800; font-size:1.05rem;
                    background:linear-gradient(135deg,#a78bfa,#60a5fa);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; letter-spacing:-0.5px;
                ">VisualMind AI</div>
                <div style="font-size:0.72rem; color:rgba(124,58,237,0.6);
                    letter-spacing:0.5px; text-transform:uppercase; margin-top:1px;">
                    Dual-Model Vision
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Theme toggle
    def _on_theme_change():
        st.session_state.theme = "dark" if st.session_state._theme_toggle else "light"

    st.toggle(
        "Dark mode",
        value=(st.session_state.theme == "dark"),
        key="_theme_toggle",
        on_change=_on_theme_change,
    )

    st.divider()

    st.markdown(
        "<div style='font-size:0.72rem; font-weight:700; text-transform:uppercase;"
        "letter-spacing:1.2px; color:rgba(124,58,237,0.55); margin-bottom:10px;'>Session Stats</div>",
        unsafe_allow_html=True,
    )
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "qa_turns" not in st.session_state:
        st.session_state.qa_turns = 0

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Tokens", f"{st.session_state.total_tokens:,}")
    with col_b:
        st.metric("Q&A Turns", st.session_state.qa_turns)

    st.divider()

    st.markdown(
        """
        <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase;
            letter-spacing:1.2px; color:rgba(124,58,237,0.55); margin-bottom:10px;">
            Models
        </div>
        <div style="display:flex; flex-direction:column; gap:8px;">
            <div style="background:rgba(124,58,237,0.07); border:1px solid rgba(124,58,237,0.15);
                border-radius:12px; padding:10px 12px;">
                <div style="font-size:0.78rem; font-weight:600; color:#a78bfa;">🤗 BLIP-large</div>
                <div style="font-size:0.72rem; color:rgba(124,58,237,0.5); margin-top:2px;">
                    HuggingFace Inference API
                </div>
            </div>
            <div style="background:rgba(37,99,235,0.07); border:1px solid rgba(37,99,235,0.15);
                border-radius:12px; padding:10px 12px;">
                <div style="font-size:0.78rem; font-weight:600; color:#60a5fa;">⚡ Llama 4 Scout</div>
                <div style="font-size:0.72rem; color:rgba(37,99,235,0.5); margin-top:2px;">
                    Groq — 17B vision model
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show a warning if keys are missing (no inputs — just a hint)
    if not groq_key or not hf_token:
        st.divider()
        missing = []
        if not groq_key:
            missing.append("GROQ_API_KEY")
        if not hf_token:
            missing.append("HF_TOKEN")
        st.warning(f"Missing env vars: {', '.join(missing)}")


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding:2.5rem 0 1.8rem 0; animation:fadeInUp 0.55s ease-out;">
        <div style="
            display:inline-flex; align-items:center; gap:8px;
            background:rgba(124,58,237,0.1); border:1px solid rgba(124,58,237,0.22);
            border-radius:100px; padding:5px 14px; margin-bottom:18px;
            font-size:0.72rem; color:rgba(124,58,237,0.75);
            letter-spacing:1.2px; text-transform:uppercase; font-weight:600;
        ">✦ &nbsp;Dual-Model Visual Intelligence</div>
        <div style="
            font-size: clamp(2.4rem, 5vw, 4rem);
            font-weight:800; letter-spacing:-3px; line-height:1.05;
            background:linear-gradient(135deg, #7c3aed 0%, #a78bfa 50%, #60a5fa 100%);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            background-clip:text;
        ">VisualMind AI</div>
        <div style="
            color:rgba(124,58,237,0.55); font-size:1.05rem; margin-top:12px;
            font-weight:400; letter-spacing:-0.2px; max-width:480px; margin-left:auto; margin-right:auto;
        ">Upload an image. Instantly understand everything in it.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Session state init ────────────────────────────────────────────────────────
_defaults: dict = {
    "image_bytes": None,
    "blip_caption": None,
    "groq_analysis": None,
    "conversation_history": [],
    "image_info": None,
    "suggested_questions": [],
    "pending_question": "",
}
for _key, _default in _defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop an image here or click to browse",
    type=["jpg", "jpeg", "png", "webp", "gif"],
    help="Max 4 MB — JPEG, PNG, WebP, GIF",
    label_visibility="visible",
)

if uploaded_file:
    image_bytes, error = validate_and_process(uploaded_file)

    if error:
        st.error(error)
    else:
        if st.session_state.image_bytes != image_bytes:
            st.session_state.image_bytes = image_bytes
            st.session_state.blip_caption = None
            st.session_state.groq_analysis = None
            st.session_state.conversation_history = []
            st.session_state.image_info = get_image_info(image_bytes)
            st.session_state.suggested_questions = []
            st.session_state.pending_question = ""

        need_blip = st.session_state.blip_caption is None
        need_groq = st.session_state.groq_analysis is None

        if need_blip or need_groq:
            if hf_token and groq_key and need_blip and need_groq:
                with st.spinner("BLIP + Llama 4 Scout analyzing in parallel…"):
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        fut_blip = executor.submit(get_blip_caption, image_bytes, hf_token)
                        fut_groq = executor.submit(get_initial_analysis, image_bytes, groq_key)
                        st.session_state.blip_caption = fut_blip.result()
                        analysis, tokens = fut_groq.result()
                        st.session_state.groq_analysis = analysis
                        st.session_state.total_tokens += tokens
            else:
                if need_blip:
                    if hf_token:
                        with st.spinner("BLIP generating caption…"):
                            st.session_state.blip_caption = get_blip_caption(image_bytes, hf_token)
                    else:
                        st.session_state.blip_caption = "⚠️ HuggingFace token not set."
                if need_groq:
                    if groq_key:
                        with st.spinner("Llama 4 Scout analyzing image…"):
                            analysis, tokens = get_initial_analysis(image_bytes, groq_key)
                            st.session_state.groq_analysis = analysis
                            st.session_state.total_tokens += tokens
                    else:
                        st.session_state.groq_analysis = "⚠️ Groq API key not set."

        # Generate suggested questions once analysis is available
        if (
            not st.session_state.suggested_questions
            and st.session_state.groq_analysis
            and not st.session_state.groq_analysis.startswith(("⚠️", "Groq API error"))
        ):
            st.session_state.suggested_questions = get_suggested_questions(
                st.session_state.groq_analysis
            )

        st.divider()

        # ── Image + Analysis ──────────────────────────────────────────────────
        col_img, col_analysis = st.columns([0.42, 0.58], gap="large")

        with col_img:
            st.markdown(
                "<div style='font-size:0.72rem; font-weight:700; text-transform:uppercase;"
                "letter-spacing:1.2px; color:rgba(124,58,237,0.55); margin-bottom:10px;'>"
                "📸 &nbsp;Uploaded Image</div>",
                unsafe_allow_html=True,
            )
            st.image(image_bytes, use_container_width=True)
            info = st.session_state.image_info
            st.caption(
                f"{info['width']} × {info['height']} px  ·  {info['size_kb']} KB  ·  {info['mode']}"
            )

        with col_analysis:
            st.markdown(
                "<div style='font-size:0.72rem; font-weight:700; text-transform:uppercase;"
                "letter-spacing:1.2px; color:rgba(124,58,237,0.55); margin-bottom:10px;'>"
                "🤖 &nbsp;AI Analysis</div>",
                unsafe_allow_html=True,
            )

            # BLIP card
            st.markdown(
                """
                <div style="
                    background:rgba(124,58,237,0.06); border:1px solid rgba(124,58,237,0.18);
                    border-radius:16px; padding:14px 16px; margin-bottom:6px;
                    animation:fadeInUp 0.45s ease-out;
                ">
                    <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase;
                        letter-spacing:1px; color:#a78bfa; margin-bottom:6px;">
                        🤗 &nbsp;HuggingFace BLIP-large — Caption
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info(st.session_state.blip_caption or "—")

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            # Groq card
            st.markdown(
                """
                <div style="
                    background:rgba(37,99,235,0.06); border:1px solid rgba(37,99,235,0.18);
                    border-radius:16px; padding:14px 16px; margin-bottom:6px;
                    animation:fadeInUp 0.55s ease-out;
                ">
                    <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase;
                        letter-spacing:1px; color:#60a5fa; margin-bottom:6px;">
                        ⚡ &nbsp;Llama 4 Scout (Groq) — Deep Analysis
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.success(st.session_state.groq_analysis or "—")

            # Action row
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            btn_dl, btn_clear, _ = st.columns([0.38, 0.35, 0.27])
            with btn_dl:
                combined_text = (
                    f"=== VisualMind AI Analysis ===\n\n"
                    f"Image: {info['width']}×{info['height']}px, {info['size_kb']} KB\n\n"
                    f"--- BLIP Caption ---\n{st.session_state.blip_caption}\n\n"
                    f"--- Llama 4 Scout Analysis ---\n{st.session_state.groq_analysis}\n\n"
                    f"--- Conversation ---\n"
                    + "\n".join(
                        f"[{m['role'].upper()}] {m['content']}"
                        for m in st.session_state.conversation_history
                    )
                )
                st.download_button(
                    "⬇ Download",
                    data=combined_text,
                    file_name="visualmind_analysis.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with btn_clear:
                if st.button("🗑 Clear Chat", use_container_width=True):
                    st.session_state.conversation_history = []
                    st.session_state.pending_question = ""
                    st.rerun()

            # Copy-friendly plain-text view
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            with st.expander("📋 Copy Analysis Text", expanded=False):
                st.code(
                    f"BLIP Caption:\n{st.session_state.blip_caption or '—'}\n\n"
                    f"Llama 4 Scout Analysis:\n{st.session_state.groq_analysis or '—'}",
                    language=None,
                )

        # ── Chat ──────────────────────────────────────────────────────────────
        st.divider()
        st.markdown(
            """
            <div style="animation:fadeInUp 0.6s ease-out;">
                <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase;
                    letter-spacing:1.2px; color:rgba(124,58,237,0.55); margin-bottom:4px;">
                    💬 &nbsp;Follow-Up Q&A
                </div>
                <div style="color:rgba(124,58,237,0.45); font-size:0.85rem; margin-bottom:16px;">
                    Ask anything about the image — objects, text, colors, context, comparisons.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Suggested questions (shown only before first Q&A turn)
        if st.session_state.suggested_questions and not st.session_state.conversation_history:
            st.markdown(
                "<div style='font-size:0.75rem; color:rgba(124,58,237,0.5); "
                "margin-bottom:8px; letter-spacing:0.2px;'>✦ Suggested questions</div>",
                unsafe_allow_html=True,
            )
            sq_cols = st.columns(len(st.session_state.suggested_questions), gap="small")
            for sq_col, sq in zip(sq_cols, st.session_state.suggested_questions):
                if sq_col.button(sq, use_container_width=True, key=f"sq_{sq[:24]}"):
                    st.session_state.pending_question = sq
                    st.rerun()
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        for message in st.session_state.conversation_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_question = st.chat_input("What would you like to know about this image?")

        active_question = user_question
        if st.session_state.pending_question:
            active_question = st.session_state.pending_question
            st.session_state.pending_question = ""

        if active_question:
            if not groq_key:
                st.warning("GROQ_API_KEY is not set — add it to your .env or Space secrets.")
            else:
                with st.chat_message("user"):
                    st.markdown(active_question)

                with st.chat_message("assistant"):
                    token_bucket: list[int] = []
                    answer = st.write_stream(
                        answer_followup_stream(
                            st.session_state.image_bytes,
                            active_question,
                            st.session_state.conversation_history,
                            groq_key,
                            token_bucket,
                        )
                    )
                    tokens = token_bucket[0] if token_bucket else 0
                    if tokens:
                        st.caption(f"Tokens used this turn: {tokens:,}")

                st.session_state.conversation_history.append({"role": "user", "content": active_question})
                st.session_state.conversation_history.append({"role": "assistant", "content": answer})
                st.session_state.total_tokens += tokens
                st.session_state.qa_turns += 1

# ── Landing (no image yet) ────────────────────────────────────────────────────
else:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.72rem; font-weight:700; text-transform:uppercase;"
        "letter-spacing:1.2px; color:rgba(124,58,237,0.55); margin-bottom:16px; text-align:center;'>"
        "What can VisualMind AI do?</div>",
        unsafe_allow_html=True,
    )

    use_cases = [
        ("🛍️", "E-Commerce", "Product tagging, quality checks, listing copy generation"),
        ("♿", "Accessibility", "Auto alt-text for websites, screen-reader descriptions"),
        ("🔍", "Research", "Charts, slides, documents, scene Q&A, OCR context"),
    ]
    cols = st.columns(3, gap="medium")
    for col, (icon, title, desc) in zip(cols, use_cases):
        with col:
            st.markdown(
                f"""
                <div style="
                    background:rgba(124,58,237,0.04); border:1px solid rgba(124,58,237,0.12);
                    border-radius:20px; padding:22px 20px; min-height:130px;
                    transition:all 0.3s ease; animation:fadeInUp 0.6s ease-out;
                    cursor:default;
                " onmouseover="this.style.borderColor='rgba(124,58,237,0.3)';
                               this.style.background='rgba(124,58,237,0.08)';
                               this.style.transform='translateY(-4px)';"
                  onmouseout="this.style.borderColor='rgba(124,58,237,0.12)';
                              this.style.background='rgba(124,58,237,0.04)';
                              this.style.transform='translateY(0)';">
                    <div style="font-size:1.8rem; margin-bottom:10px;">{icon}</div>
                    <div style="font-weight:700; color:#7c3aed; font-size:0.95rem;
                        margin-bottom:7px; letter-spacing:-0.3px;">{title}</div>
                    <div style="color:rgba(124,58,237,0.55); font-size:0.83rem; line-height:1.6;">
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
