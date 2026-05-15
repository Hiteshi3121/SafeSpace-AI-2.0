"""
interfaces/streamlit_ui/app.py
SafeSpace AI — Streamlit Web Interface with text, image, and audio support.
HOW TO RUN:
    streamlit run interfaces/streamlit_ui/app.py
"""

import asyncio
import concurrent.futures
import sys
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.schemas import ChatRequest, ChatResponse, Intent, MessageType
from core.engine import handle_request

st.set_page_config(
    page_title="SafeSpace AI",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&family=Fraunces:ital,wght@0,300;0,400;1,300&display=swap');

    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; background: #0d1117; }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stAppViewBlockContainer"] { padding-top: 0 !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #111820 !important; border-right: 1px solid #1a2d1a; }
    [data-testid="stSidebar"] * { color: #8aaa8a !important; }
    [data-testid="stSidebar"] h3 {
        font-size: 0.72rem !important; font-weight: 600 !important;
        letter-spacing: 0.12em !important; text-transform: uppercase !important;
        color: #3a6a3a !important; margin-bottom: 0.5rem !important;
    }
    section[data-testid="stSidebar"] > div { padding-top: 2rem !important; }

    /* Header */
    .safe-header { text-align: center; padding: 2rem 0 1.2rem 0; border-bottom: 1px solid #1a2d1a; margin-bottom: 1rem; }
    .safe-header h1 { font-family: 'Fraunces', serif; font-size: 2rem; font-weight: 300; color: #e0f0e0; margin: 0; }
    .safe-header h1 span { color: #4a9e4a; font-style: italic; }
    .safe-header p { color: #3a5a3a; font-size: 0.75rem; margin-top: 0.4rem; letter-spacing: 0.08em; text-transform: uppercase; }

    /* Messages */
    .msg-user {
        background: linear-gradient(135deg, #182e18, #1c3a1c);
        color: #e0f0e0; padding: 0.85rem 1.15rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0 0.15rem 4rem;
        font-size: 0.92rem; line-height: 1.55;
        border: 1px solid #254025;
        box-shadow: 0 2px 12px rgba(0,0,0,0.4);
    }
    .msg-bot {
        background: #111a11; color: #c8e0c8;
        padding: 0.85rem 1.15rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 4rem 0.15rem 0;
        font-size: 0.92rem; line-height: 1.65;
        border: 1px solid #1a2d1a;
        box-shadow: 0 2px 12px rgba(0,0,0,0.4);
    }
    .msg-meta { font-size: 0.67rem; color: #2e4a2e; margin: 0 0 0.8rem 0.4rem; }
    .msg-meta-right { text-align: right; padding-right: 0.4rem; margin: 0 0 0.8rem 0; font-size: 0.67rem; color: #2e4a2e; }

    /* Badges */
    .badge { display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 0.64rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }
    .badge-medical { background: #0a1525; color: #4a90f5; border: 1px solid #1a3a6a; }
    .badge-therapy { background: #150a25; color: #9a6af5; border: 1px solid #3a1a6a; }
    .badge-mixed   { background: #201508; color: #f59a3a; border: 1px solid #5a3a1a; }
    .badge-unknown { background: #151515; color: #4a4a4a; border: 1px solid #252525; }

    /* Emergency */
    .emergency-alert { background: #180808; border: 1px solid #4a1010; border-radius: 10px; padding: 0.85rem 1.1rem; color: #f07070; font-size: 0.87rem; margin: 0.5rem 0; }

    /* Divider */
    .input-divider { border: none; border-top: 1px solid #1a2d1a; margin: 1rem 0 0.8rem 0; }

    /* Buttons */
    .stButton > button {
        border-radius: 10px !important; background: #182e18 !important;
        color: #8aca8a !important; border: 1px solid #254025 !important;
        font-family: 'Outfit', sans-serif !important; font-size: 0.82rem !important;
        font-weight: 500 !important; transition: all 0.18s !important;
    }
    .stButton > button:hover { background: #1e3a1e !important; color: #b0e0b0 !important; border-color: #3a6a3a !important; }

    /* Text input */
    .stTextInput > div > div > input {
        background: #0d1a0d !important; color: #c8e0c8 !important;
        border: 1px solid #1a2d1a !important; border-radius: 10px !important;
        font-family: 'Outfit', sans-serif !important; font-size: 0.9rem !important;
        caret-color: #4a9e4a;
    }
    .stTextInput > div > div > input::placeholder { color: #253525 !important; }
    .stTextInput > div > div > input:focus { border-color: #2a5a2a !important; box-shadow: 0 0 0 2px rgba(42,90,42,0.15) !important; }

    /* File uploader */
    [data-testid="stFileUploader"] { background: #0d1a0d !important; border: 1px dashed #1a3a1a !important; border-radius: 10px !important; }
    [data-testid="stFileUploader"] label { color: #3a6a3a !important; font-size: 0.72rem !important; font-weight: 600 !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] { font-size: 0.7rem !important; color: #253525 !important; }

    /* mic receiver removed — using st.audio_input instead */

    /* Scrollbar */
    ::-webkit-scrollbar { width: 3px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #1a3a1a; border-radius: 3px; }

    /* Spinner */
    .stSpinner > div { border-top-color: #4a9e4a !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = f"web_{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

# ── Init SQLite DB ────────────────────────────────────────────────────────────
if "db_initialized" not in st.session_state:
    def _init_db():
        import os
        db_path = os.getenv("SQLITE_DB_PATH", "./data/safespace.db")
        dir_path = os.path.dirname(db_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            from memory.store import init_db
            loop.run_until_complete(init_db())
        finally:
            loop.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        ex.submit(_init_db).result(timeout=30)
    st.session_state.db_initialized = True


def run_async(coro):
    def _run(c):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(c)
        finally:
            loop.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_run, coro).result(timeout=120)


def clean_response(text: str) -> str:
    import re
    text = re.sub(r'<function=\w+>.*?(?:</function>|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'\{"location":.*?\}', '', text, flags=re.DOTALL)
    return text.strip()


def intent_badge(intent) -> str:
    cls_map = {
        Intent.MEDICAL: "badge-medical",
        Intent.THERAPY: "badge-therapy",
        Intent.MIXED:   "badge-mixed",
        Intent.UNKNOWN: "badge-unknown",
    }
    cls = cls_map.get(intent, "badge-unknown")
    label = intent.value if intent else "UNKNOWN"
    return f'<span class="badge {cls}">{label}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 SafeSpace AI")
    st.markdown('<hr class="input-divider">', unsafe_allow_html=True)
    st.markdown("**Session ID**")
    st.code(st.session_state.session_id, language=None)
    st.markdown("**What I can help with:**")
    st.markdown("""
- 🩺 Medical symptoms & guidance
- 🧠 Anxiety, stress & emotional support
- 🖼️ Medical image analysis
- 🎙️ Voice message support
- 🗺️ Finding therapists near you
    """)
    st.markdown('<hr class="input-divider">', unsafe_allow_html=True)
    st.markdown("**Crisis Resources (India)**")
    st.markdown("""
- iCall: **9152987821**
- Vandrevala: **1860-2662-345**
- Emergency: **112**
    """)
    st.markdown('<hr class="input-divider">', unsafe_allow_html=True)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_input = None
        async def _clear():
            from memory.store import clear_session
            await clear_session(st.session_state.session_id)
        run_async(_clear())
        st.rerun()
    st.markdown('<hr class="input-divider">', unsafe_allow_html=True)
    st.caption("⚠️ SafeSpace is an AI assistant. Always consult a real doctor or therapist for medical advice.")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="safe-header">
    <h1>🌿 Safe<span>Space</span></h1>
    <p>Your AI medical &amp; mental health companion</p>
</div>
""", unsafe_allow_html=True)


# ── Process pending input ─────────────────────────────────────────────────────
if st.session_state.pending_input:
    pending = st.session_state.pending_input
    st.session_state.pending_input = None

    with st.spinner("SafeSpace is thinking..."):
        try:
            request = ChatRequest(
                user_id=st.session_state.session_id,
                message_type=pending["type"],
                text=pending.get("text"),
                audio_bytes=pending.get("audio_bytes"),
                image_bytes=pending.get("image_bytes"),
            )
            response: ChatResponse = run_async(handle_request(request))
            response_text = clean_response(response.text)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "intent": response.intent,
                "escalated": response.escalated,
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I'm sorry, something went wrong. Please try again.\n\n_(Error: {str(e)[:120]})_",
                "intent": Intent.UNKNOWN,
                "escalated": False,
            })


# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="msg-meta msg-meta-right">You</div>', unsafe_allow_html=True)
    else:
        if msg.get("escalated"):
            st.markdown("""
            <div class="emergency-alert">
                🚨 <strong>Emergency alert triggered.</strong>
                Please call <strong>112</strong> if you need immediate help.
            </div>
            """, unsafe_allow_html=True)
        badge = intent_badge(msg.get("intent", Intent.UNKNOWN))
        st.markdown(f'<div class="msg-bot">{msg["content"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="msg-meta">SafeSpace · {badge}</div>', unsafe_allow_html=True)


# ── Input area ────────────────────────────────────────────────────────────────
st.markdown('<hr class="input-divider">', unsafe_allow_html=True)

# ── Top row: Speak | Upload Audio | Upload Image ──────────────────────────────
# Layout matches user sketch: [🎙 Speak] [☁️ Upload Audio] [📷 Upload Image]
mic_col, audio_col, image_col = st.columns([1, 1, 1])

# 🎙 SPEAK — native Streamlit mic (works in all browsers, no JS needed)
with mic_col:
    audio_input = st.audio_input(
        "🎙️ Speak",
        key="mic_recorder",
    )
    if audio_input is not None:
        mic_key = f"mic_{audio_input.name}_{audio_input.size}"
        if mic_key not in st.session_state:
            st.session_state[mic_key] = True
            audio_bytes = audio_input.read()
            st.session_state.messages.append({
                "role": "user",
                "content": "🎙️ Voice message recorded",
            })
            st.session_state.pending_input = {
                "type": MessageType.AUDIO,
                "audio_bytes": audio_bytes,
                "text": "recorded.wav",
            }
            st.rerun()

# ☁️ UPLOAD AUDIO
with audio_col:
    uploaded_audio = st.file_uploader(
        "☁️ Upload Audio",
        type=["ogg", "mp3", "wav", "m4a", "webm"],
        key="audio_upload",
        label_visibility="visible",
    )

# 📷 UPLOAD IMAGE
with image_col:
    uploaded_image = st.file_uploader(
        "📷 Upload Image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
        label_visibility="visible",
    )

# ── Bottom row: text input + send ────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        label="message",
        placeholder="Type your message...",
        label_visibility="collapsed",
        key="chat_input",
    )
with col2:
    send_clicked = st.button("➤", use_container_width=True)


# ── Handle send ───────────────────────────────────────────────────────────────

# Text send
if send_clicked and user_input.strip():
    user_text = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.pending_input = {
        "type": MessageType.TEXT,
        "text": user_text,
    }
    st.rerun()

# Image send — use filename as processed key to avoid re-triggering
if uploaded_image is not None:
    img_key = f"img_{uploaded_image.name}_{uploaded_image.size}"
    if img_key not in st.session_state:
        st.session_state[img_key] = True
        image_bytes = uploaded_image.read()
        st.session_state.messages.append({
            "role": "user",
            "content": f"📷 Image uploaded: {uploaded_image.name}",
        })
        st.session_state.pending_input = {
            "type": MessageType.IMAGE,
            "image_bytes": image_bytes,
        }
        st.rerun()

# Audio send — same pattern
if uploaded_audio is not None:
    aud_key = f"aud_{uploaded_audio.name}_{uploaded_audio.size}"
    if aud_key not in st.session_state:
        st.session_state[aud_key] = True
        audio_bytes = uploaded_audio.read()
        st.session_state.messages.append({
            "role": "user",
            "content": f"🎙️ Voice note uploaded: {uploaded_audio.name}",
        })
        st.session_state.pending_input = {
            "type": MessageType.AUDIO,
            "audio_bytes": audio_bytes,
            "text": uploaded_audio.name,  # Pass filename for format detection
        }
        st.rerun()