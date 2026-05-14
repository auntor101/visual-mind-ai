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


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap');

/* ── Base ── */
html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif !important;
    background: #07070d !important;
    color: #f1f5f9 !important;
}
* { box-sizing: border-box; }

/* Hide Streamlit chrome */
#MainMenu { display: none !important; }
footer { display: none !important; }
header { display: none !important; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.35); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,0.6); }

/* ── Keyframes ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(22px); }
    to   { opacity: 1; transform: translateY(0);    }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes gradientFlow {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}
@keyframes glowPulse {
    0%,100% { box-shadow: 0 0 24px rgba(124,58,237,0.08); }
    50%      { box-shadow: 0 0 48px rgba(124,58,237,0.22); }
}
@keyframes borderGlow {
    0%,100% { border-color: rgba(124,58,237,0.3); }
    50%      { border-color: rgba(124,58,237,0.7); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-16px); }
    to   { opacity: 1; transform: translateX(0);     }
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(16px); }
    to   { opacity: 1; transform: translateX(0);    }
}
@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.96); }
    to   { opacity: 1; transform: scale(1);    }
}

/* ── Main block ── */
.main .block-container {
    animation: fadeIn 0.45s ease-out;
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1380px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(8, 8, 18, 0.97) !important;
    border-right: 1px solid rgba(255,255,255,0.055) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: rgba(248,250,252,0.6) !important;
    font-size: 0.85rem !important;
    line-height: 1.65 !important;
}
[data-testid="stSidebar"] label {
    color: rgba(248,250,252,0.65) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px !important;
}

/* ── Headings ── */
h1 {
    font-weight: 800 !important;
    letter-spacing: -2.5px !important;
    line-height: 1.05 !important;
}
h2, h3 {
    font-weight: 700 !important;
    letter-spacing: -0.6px !important;
    color: #f1f5f9 !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent) !important;
    margin: 1.4rem 0 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    animation: fadeInUp 0.55s ease-out;
}
[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.018) !important;
    border: 1.5px dashed rgba(124,58,237,0.38) !important;
    border-radius: 22px !important;
    padding: 2.8rem 2rem !important;
    transition: background 0.25s ease, border-color 0.25s ease, transform 0.25s ease, box-shadow 0.25s ease !important;
    animation: glowPulse 3.5s ease-in-out infinite;
}
[data-testid="stFileUploader"] section:hover {
    background: rgba(124,58,237,0.055) !important;
    border-color: rgba(124,58,237,0.65) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 44px rgba(124,58,237,0.14) !important;
}
[data-testid="stFileUploader"] section > div > span {
    color: rgba(248,250,252,0.65) !important;
    font-size: 0.95rem !important;
}
[data-testid="stFileUploader"] section > div small {
    color: rgba(248,250,252,0.35) !important;
    font-size: 0.78rem !important;
}

/* ── Primary buttons ── */
.stButton > button {
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
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124,58,237,0.45) !important;
    filter: brightness(1.08) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 3px 12px rgba(124,58,237,0.3) !important;
}

/* ── Password / Text inputs ── */
[data-testid="stPasswordInput"] input,
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.038) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease !important;
}
[data-testid="stPasswordInput"] input:focus,
.stTextInput > div > div > input:focus {
    border-color: rgba(124,58,237,0.55) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.13) !important;
    background: rgba(255,255,255,0.055) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.028) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 18px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(124,58,237,0.48) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 18px !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    margin: 0.45rem 0 !important;
    animation: fadeInUp 0.3s ease-out !important;
    transition: border-color 0.2s ease, background 0.2s ease !important;
}
[data-testid="stChatMessage"]:hover {
    border-color: rgba(124,58,237,0.18) !important;
    background: rgba(255,255,255,0.035) !important;
}

/* ── Alert boxes ── */
[data-testid="stAlert"] {
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    animation: fadeInUp 0.35s ease-out !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
}
/* info */
[data-testid="stAlert"][data-baseweb="notification"][kind="info"],
div[data-testid="stAlert"] {
    background: rgba(37,99,235,0.07) !important;
    border: 1px solid rgba(37,99,235,0.2) !important;
    color: rgba(248,250,252,0.88) !important;
}
/* success */
.element-container div[data-testid="stAlert"][kind="success"] {
    background: rgba(5,150,105,0.07) !important;
    border: 1px solid rgba(5,150,105,0.2) !important;
}
/* warning */
[data-testid="stAlert"][kind="warning"] {
    background: rgba(217,119,6,0.07) !important;
    border: 1px solid rgba(217,119,6,0.22) !important;
    color: #fbbf24 !important;
}
/* error */
[data-testid="stAlert"][kind="error"] {
    background: rgba(220,38,38,0.07) !important;
    border: 1px solid rgba(220,38,38,0.22) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 16px !important;
    padding: 1rem 1.1rem !important;
    transition: border-color 0.2s ease, background 0.2s ease, transform 0.2s ease !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(124,58,237,0.28) !important;
    background: rgba(124,58,237,0.048) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
[data-testid="stMetricLabel"] > div {
    color: rgba(248,250,252,0.42) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.1px !important;
}
[data-testid="stMetricDelta"] { display: none !important; }

/* ── Image ── */
[data-testid="stImage"] > img {
    border-radius: 18px !important;
    box-shadow: 0 8px 48px rgba(0,0,0,0.45) !important;
    transition: transform 0.35s ease, box-shadow 0.35s ease !important;
    animation: scaleIn 0.5s ease-out;
}
[data-testid="stImage"] > img:hover {
    transform: scale(1.015) !important;
    box-shadow: 0 14px 64px rgba(0,0,0,0.55) !important;
}

/* ── Captions ── */
[data-testid="stCaptionContainer"] p {
    color: rgba(248,250,252,0.38) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1px !important;
}

/* ── Markdown paragraphs ── */
.stMarkdown p {
    color: rgba(248,250,252,0.78) !important;
    line-height: 1.72 !important;
    font-size: 0.93rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 16px !important;
}
[data-testid="stExpander"] summary {
    color: rgba(248,250,252,0.8) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: rgba(255,255,255,0.04) !important;
    color: rgba(248,250,252,0.75) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: rgba(124,58,237,0.1) !important;
    border-color: rgba(124,58,237,0.35) !important;
    color: #f1f5f9 !important;
    transform: translateY(-1px) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    border-top-color: #7c3aed !important;
}

/* ── Column animations ── */
[data-testid="column"]:nth-child(1) { animation: slideInLeft  0.5s ease-out; }
[data-testid="column"]:nth-child(2) { animation: slideInRight 0.5s ease-out; }

/* ── Sidebar HR ── */
[data-testid="stSidebar"] hr {
    background: rgba(255,255,255,0.05) !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="
            display:flex; align-items:center; gap:10px;
            margin-bottom:4px;
        ">
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
                <div style="font-size:0.72rem; color:rgba(248,250,252,0.38);
                    letter-spacing:0.5px; text-transform:uppercase; margin-top:1px;">
                    Dual-Model Vision
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        "<div style='font-size:0.72rem; font-weight:700; text-transform:uppercase;"
        "letter-spacing:1.2px; color:rgba(248,250,252,0.35); margin-bottom:8px;'>API Keys</div>",
        unsafe_allow_html=True,
    )
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        value=get_api_key("GROQ_API_KEY"),
        help="Free at console.groq.com — powers Llama 4 Scout analysis & Q&A",
        placeholder="gsk_...",
    )
    hf_token = st.text_input(
        "HuggingFace Token",
        type="password",
        value=get_api_key("HF_TOKEN"),
        help="Free at huggingface.co/settings/tokens — powers BLIP captioning",
        placeholder="hf_...",
    )

    st.divider()

    st.markdown(
        "<div style='font-size:0.72rem; font-weight:700; text-transform:uppercase;"
        "letter-spacing:1.2px; color:rgba(248,250,252,0.35); margin-bottom:10px;'>Session Stats</div>",
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
            letter-spacing:1.2px; color:rgba(248,250,252,0.35); margin-bottom:10px;">
            Models
        </div>
        <div style="display:flex; flex-direction:column; gap:8px;">
            <div style="background:rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.07);
                border-radius:12px; padding:10px 12px;">
                <div style="font-size:0.78rem; font-weight:600; color:#a78bfa;">🤗 BLIP-large</div>
                <div style="font-size:0.72rem; color:rgba(248,250,252,0.4); margin-top:2px;">
                    HuggingFace Inference API
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.07);
                border-radius:12px; padding:10px 12px;">
                <div style="font-size:0.78rem; font-weight:600; color:#60a5fa;">⚡ Llama 4 Scout</div>
                <div style="font-size:0.72rem; color:rgba(248,250,252,0.4); margin-top:2px;">
                    Groq — 17B vision model
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding:2.5rem 0 1.8rem 0; animation:fadeInUp 0.55s ease-out;">
        <div style="
            display:inline-flex; align-items:center; gap:8px;
            background:rgba(124,58,237,0.1); border:1px solid rgba(124,58,237,0.22);
            border-radius:100px; padding:5px 14px; margin-bottom:18px;
            font-size:0.72rem; color:rgba(248,250,252,0.6);
            letter-spacing:1.2px; text-transform:uppercase; font-weight:600;
        ">✦ &nbsp;Dual-Model Visual Intelligence</div>
        <div style="
            font-size: clamp(2.4rem, 5vw, 4rem);
            font-weight:800; letter-spacing:-3px; line-height:1.05;
            background:linear-gradient(135deg, #f8fafc 0%, #c4b5fd 50%, #93c5fd 100%);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            background-clip:text;
        ">VisualMind AI</div>
        <div style="
            color:rgba(248,250,252,0.42); font-size:1.05rem; margin-top:12px;
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
                        st.session_state.blip_caption = "⚠️ HuggingFace token not provided."
                if need_groq:
                    if groq_key:
                        with st.spinner("Llama 4 Scout analyzing image…"):
                            analysis, tokens = get_initial_analysis(image_bytes, groq_key)
                            st.session_state.groq_analysis = analysis
                            st.session_state.total_tokens += tokens
                    else:
                        st.session_state.groq_analysis = "⚠️ Groq API key not provided."

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
                "letter-spacing:1.2px; color:rgba(248,250,252,0.35); margin-bottom:10px;'>"
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
                "letter-spacing:1.2px; color:rgba(248,250,252,0.35); margin-bottom:10px;'>"
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

            # Action row: Download + Clear
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

            # Copy-friendly plain-text view of the full analysis
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
                    letter-spacing:1.2px; color:rgba(248,250,252,0.35); margin-bottom:4px;">
                    💬 &nbsp;Follow-Up Q&A
                </div>
                <div style="color:rgba(248,250,252,0.42); font-size:0.85rem; margin-bottom:16px;">
                    Ask anything about the image — objects, text, colors, context, comparisons.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Suggested questions (shown only before first Q&A turn)
        if st.session_state.suggested_questions and not st.session_state.conversation_history:
            st.markdown(
                "<div style='font-size:0.75rem; color:rgba(248,250,252,0.38); "
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

        # Pending question wins (set by a suggested-question button click)
        active_question = user_question
        if st.session_state.pending_question:
            active_question = st.session_state.pending_question
            st.session_state.pending_question = ""

        if active_question:
            if not groq_key:
                st.warning("Groq API key required for Q&A — add it in the sidebar.")
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
        "letter-spacing:1.2px; color:rgba(248,250,252,0.35); margin-bottom:16px; text-align:center;'>"
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
                    background:rgba(255,255,255,0.022); border:1px solid rgba(255,255,255,0.07);
                    border-radius:20px; padding:22px 20px; min-height:130px;
                    transition:all 0.3s ease; animation:fadeInUp 0.6s ease-out;
                    cursor:default;
                " onmouseover="this.style.borderColor='rgba(124,58,237,0.3)';
                               this.style.background='rgba(124,58,237,0.05)';
                               this.style.transform='translateY(-4px)';"
                  onmouseout="this.style.borderColor='rgba(255,255,255,0.07)';
                              this.style.background='rgba(255,255,255,0.022)';
                              this.style.transform='translateY(0)';">
                    <div style="font-size:1.8rem; margin-bottom:10px;">{icon}</div>
                    <div style="font-weight:700; color:#f1f5f9; font-size:0.95rem;
                        margin-bottom:7px; letter-spacing:-0.3px;">{title}</div>
                    <div style="color:rgba(248,250,252,0.48); font-size:0.83rem; line-height:1.6;">
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
