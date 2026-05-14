"""
hf_app.py  —  Hugging Face Spaces entry point

HF Spaces runs:  streamlit run hf_app.py --server.port 7860

WHY THIS FILE EXISTS:
  Our real Streamlit UI is at interfaces/streamlit_ui/app.py and
  uses imports relative to the project root (e.g. from core.engine import ...).
  This file sets up sys.path correctly before Streamlit boots,
  then imports and runs the real app in the same process.

  We use `exec` + file read (not runpy) to avoid Streamlit
  re-registering set_page_config in a sub-module context,
  which would throw a StreamlitAPIException.

SECRETS to add in HF Space Settings → Repository Secrets:
  GROQ_API_KEY          ← required
  GOOGLE_MAPS_API_KEY   ← required for therapist finder
  LANGSMITH_API_KEY     ← optional
  LANGSMITH_PROJECT     ← optional (default: safespace-ai)
  TWILIO_ACCOUNT_SID    ← optional (WhatsApp interface)
  TWILIO_AUTH_TOKEN     ← optional
"""

import sys
import os
from pathlib import Path

# ── 1. Add project root to sys.path ──────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── 2. Create data dir for SQLite ─────────────────────────────────────────────
os.makedirs(str(ROOT / "data"), exist_ok=True)

# ── 3. Inject env vars so CrewAI/LiteLLM can find them ───────────────────────
# pydantic-settings loads .env into Settings() but does NOT write to os.environ.
# CrewAI reads os.environ directly, so we bridge them here.
try:
    from core.config import get_settings
    _s = get_settings()
    if _s.groq_api_key:
        os.environ.setdefault("GROQ_API_KEY", _s.groq_api_key)
    if _s.google_maps_api_key:
        os.environ.setdefault("GOOGLE_MAPS_API_KEY", _s.google_maps_api_key)
    if _s.langsmith_api_key:
        os.environ.setdefault("LANGCHAIN_API_KEY", _s.langsmith_api_key)
        os.environ.setdefault("LANGSMITH_API_KEY", _s.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", _s.langsmith_project)
except Exception as e:
    print(f"[hf_app] Warning: could not load settings: {e}")

# ── 4. Execute the real Streamlit app in this process ────────────────────────
_app_path = ROOT / "interfaces" / "streamlit_ui" / "app.py"
with open(_app_path, "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), str(_app_path), "exec"), {"__name__": "__main__"})