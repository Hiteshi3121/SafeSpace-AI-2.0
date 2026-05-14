"""
hf_app.py — Hugging Face Spaces entry point

HF runs: streamlit run hf_app.py --server.port 7860

FIX: When using exec() to run the child app, __file__ is not defined
in the exec'd code's namespace. We pass it explicitly via globals dict.
"""

import sys
import os
from pathlib import Path

# ── 1. Project root is always /app on HF Docker Spaces ───────────────────────
ROOT = Path("/app")
sys.path.insert(0, str(ROOT))

# ── 2. Create data dir for SQLite ─────────────────────────────────────────────
os.makedirs(str(ROOT / "data"), exist_ok=True)

# ── 3. Inject API keys into os.environ so CrewAI/LiteLLM can find them ───────
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

# ── 4. Execute the real Streamlit app ─────────────────────────────────────────
# We use exec() so Streamlit sees it as the main script.
# IMPORTANT: pass __file__ explicitly — exec() doesn't set it automatically,
# causing NameError in any code that uses Path(__file__).
_app_path = ROOT / "interfaces" / "streamlit_ui" / "app.py"
with open(_app_path, "r", encoding="utf-8") as _f:
    exec(
        compile(_f.read(), str(_app_path), "exec"),
        {
            "__name__": "__main__",
            "__file__": str(_app_path),   # ← THIS was missing, caused the error
        }
    )