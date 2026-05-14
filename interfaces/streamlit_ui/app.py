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
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .safe-header { text-align: center; padding: 2rem 0 1rem 0; }
    .safe-header h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem; color: #1a2e1a; margin: 0; letter-spacing: -0.5px;
    }
    .safe-header p { color: #5a7a5a; font-size: 0.95rem; margin-top: 0.3rem; font-weight: 300; }
    .msg-user {
        background: #1a2e1a; color: #f0f7f0; padding: 0.85rem 1.1rem;
        border-radius: 18px 18px 4px 18px; margin: 0.4rem 0 0.1rem 3rem;
        font-size: 0.95rem; line-height: 1.5;
    }
    .msg-bot {
        background: #f0f7f0; color: #1a2e1a; padding: 0.85rem 1.1rem;
        border-radius: 18px 18px 18px 4px; margin: 0.4rem 3rem 0.1rem 0;
        font-size: 0.95rem; line-height: 1.6; border: 1px solid #d4e8d4;
    }
    .msg-meta { font-size: 0.72rem; color: #8aaa8a; margin: 0 0 0.6rem 0.3rem; }
    .msg-meta-right { text-align: right; padding-right: 0.3rem; margin: 0 0 0.6rem 0; }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 0.7rem; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase;
    }
    .badge-medical { background: #e8f0ff; color: #2a4ab0; }
    .badge-therapy { background: #f0e8ff; color: #5a2ab0; }
    .badge-mixed   { background: #fff0e8; color: #b05a2a; }
    .badge-unknown { background: #f0f0f0; color: #666; }
    .emergency-alert {
        background: #fff0f0; border: 1px solid #ffb3b3; border-radius: 10px;
        padding: 0.8rem 1rem; color: #cc0000; font-size: 0.9rem; margin: 0.5rem 0;
    }
    .upload-label { font-size: 0.8rem; color: #5a7a5a; margin-bottom: 0.3rem; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    /* Sidebar always visible fix */
    [data-testid="stSidebar"] { min-width: 260px !important; max-width: 320px !important; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
    /* Center the main content even in wide layout */
    .main .block-container { max-width: 750px; margin: 0 auto; padding-top: 1rem; }
    .stButton > button {
        border-radius: 24px; background: #1a2e1a; color: white; border: none;
        padding: 0.5rem 1.5rem; font-family: 'DM Sans', sans-serif; font-weight: 500;
    }
    .stButton > button:hover { background: #2d4f2d; }
    [data-testid="stSidebar"] { background: #f5fbf5; }
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
    st.markdown("---")
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
    st.markdown("---")
    st.markdown("**Crisis Resources (India)**")
    st.markdown("""
- iCall: **9152987821**
- Vandrevala: **1860-2662-345**
- Emergency: **112**
    """)
    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_input = None
        async def _clear():
            from memory.store import clear_session
            await clear_session(st.session_state.session_id)
        run_async(_clear())
        st.rerun()
    st.markdown("---")
    st.caption("⚠️ SafeSpace is an AI assistant. Always consult a real doctor or therapist for medical advice.")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="safe-header">
    <h1>🌿 SafeSpace</h1>
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
st.markdown("---")

# Text input row
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        label="message",
        placeholder="How are you feeling today?",
        label_visibility="collapsed",
        key="chat_input",
    )
with col2:
    send_clicked = st.button("Send", use_container_width=True)

# Media upload row
col3, col4 = st.columns(2)
with col3:
    uploaded_image = st.file_uploader(
        "📷 Upload medical image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
        label_visibility="visible",
    )
with col4:
    uploaded_audio = st.file_uploader(
        "🎙️ Upload voice note",
        type=["ogg", "mp3", "wav", "m4a", "webm"],
        key="audio_upload",
        label_visibility="visible",
    )


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