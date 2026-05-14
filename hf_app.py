"""
hf_app.py — Hugging Face Spaces entry point (Option A: FastAPI + Streamlit together)

ARCHITECTURE:
=============
HF Docker Space only exposes ONE port externally (7860).
We run TWO servers inside the container:

  Thread 1 (main):   Streamlit on port 7860  ← web users browse here
  Thread 2 (daemon): FastAPI  on port 8000   ← Twilio calls /whatsapp/webhook here
  Thread 3 (daemon): Nginx    on port 7860   ← reverse proxies both

Wait — that's complex. Simpler approach:
  We embed FastAPI INSIDE Streamlit's process using a background thread.
  Streamlit runs on 7860 (what users see).
  FastAPI runs on 8000 (internal).
  We add a small HTTP forwarder so HF's public 7860 also handles /whatsapp/webhook.

ACTUALLY simplest: Run FastAPI on 7860, serve Streamlit as a sub-process.
But Streamlit needs its own port...

REAL SOLUTION:
==============
Run FastAPI on port 7860 (HF's required port).
Mount the Streamlit app AT a sub-path using reverse proxy.

No — even simpler:
Run BOTH in one process using threading:
  - FastAPI (uvicorn) on port 7860 — handles /whatsapp/webhook, /health
  - FastAPI also serves a route that proxies to Streamlit on 8501
  - Streamlit runs on 8501 internally

This way:
  https://hiteshiaglawe0505-safespace-ai.hf.space/           → Streamlit UI
  https://hiteshiaglawe0505-safespace-ai.hf.space/whatsapp/webhook  → Twilio webhook
  https://hiteshiaglawe0505-safespace-ai.hf.space/health     → health check

TWILIO WEBHOOK URL to set:
  https://hiteshiaglawe0505-safespace-ai.hf.space/whatsapp/webhook
"""

import sys
import os
import threading
import time
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | HF-App | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Project root setup ────────────────────────────────────────────────────────
ROOT = Path("/app")
sys.path.insert(0, str(ROOT))
os.makedirs(str(ROOT / "data"), exist_ok=True)

# ── Inject secrets into environment ──────────────────────────────────────────
try:
    from core.config import get_settings
    _s = get_settings()
    env_map = {
        "GROQ_API_KEY":         _s.groq_api_key,
        "GOOGLE_MAPS_API_KEY":  _s.google_maps_api_key,
        "LANGCHAIN_API_KEY":    _s.langsmith_api_key,
        "LANGSMITH_API_KEY":    _s.langsmith_api_key,
        "LANGCHAIN_PROJECT":    _s.langsmith_project,
        "LANGCHAIN_TRACING_V2": "true" if _s.langsmith_api_key else "false",
        "GROQ_API_KEY":         _s.groq_api_key,
    }
    for k, v in env_map.items():
        if v:
            os.environ.setdefault(k, v)
    logger.info("Secrets injected: GROQ=%s GMAPS=%s TWILIO=%s",
                bool(_s.groq_api_key),
                bool(_s.google_maps_api_key),
                bool(_s.twilio_configured))
except Exception as e:
    logger.warning("Could not load settings: %s", e)


# ── Step 1: Start Streamlit on port 8501 (internal) ──────────────────────────
def start_streamlit():
    """Run Streamlit as a subprocess on internal port 8501."""
    streamlit_app = str(ROOT / "interfaces" / "streamlit_ui" / "app.py")
    cmd = [
        sys.executable, "-m", "streamlit", "run", streamlit_app,
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ]
    logger.info("Starting Streamlit on port 8501...")
    # Pass all environment variables including the injected secrets
    proc = subprocess.Popen(cmd, env=os.environ.copy())
    return proc


# ── Step 2: Start FastAPI + reverse proxy on port 7860 (HF's public port) ────
def create_combined_app():
    """
    Creates a FastAPI app that:
    1. Handles all WhatsApp webhook routes (/whatsapp/webhook)
    2. Proxies everything else to Streamlit on port 8501
    """
    import asyncio
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse, JSONResponse, Response
    from fastapi.middleware.cors import CORSMiddleware

    # Import the real FastAPI app to get all its routes
    # We do this by importing app.py's router registrations
    from core.config import get_settings
    from memory.store import init_db
    from interfaces.whatsapp.webhook import router as whatsapp_router

    settings = get_settings()

    combined = FastAPI(
        title="SafeSpace AI",
        description="AI Medical & Mental Health Assistant",
        version="2.0.0",
    )

    combined.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── WhatsApp webhook routes ───────────────────────────────────────────────
    combined.include_router(whatsapp_router, prefix="/whatsapp", tags=["WhatsApp"])

    # ── Health check ──────────────────────────────────────────────────────────
    @combined.get("/health")
    async def health():
        return {
            "status": "ok",
            "version": "2.0.0",
            "twilio": settings.twilio_configured,
            "maps": settings.maps_configured,
            "streamlit": "http://localhost:8501",
            "webhook": "https://hiteshiaglawe0505-safespace-ai.hf.space/whatsapp/webhook",
        }

    # ── Proxy everything else → Streamlit on 8501 ─────────────────────────────
    # This means visiting the HF Space URL shows the Streamlit UI
    @combined.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    )
    async def proxy_to_streamlit(request: Request, path: str = ""):
        """
        Forward all non-webhook requests to Streamlit running on port 8501.
        This makes the Streamlit UI accessible at the HF Space URL.
        """
        target_url = f"http://localhost:8501/{path}"
        params = str(request.url.query)
        if params:
            target_url += f"?{params}"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Forward the request
                proxy_response = await client.request(
                    method=request.method,
                    url=target_url,
                    headers={
                        k: v for k, v in request.headers.items()
                        if k.lower() not in ("host", "content-length")
                    },
                    content=await request.body(),
                    follow_redirects=True,
                )

                # Stream response back
                return Response(
                    content=proxy_response.content,
                    status_code=proxy_response.status_code,
                    headers=dict(proxy_response.headers),
                    media_type=proxy_response.headers.get("content-type"),
                )
        except httpx.ConnectError:
            # Streamlit not ready yet — return loading page
            return Response(
                content=_loading_html(),
                media_type="text/html",
                status_code=503,
            )
        except Exception as e:
            logger.warning("Proxy error for /%s: %s", path, e)
            return Response(
                content=_loading_html(),
                media_type="text/html",
                status_code=503,
            )

    # ── DB init on startup ────────────────────────────────────────────────────
    @combined.on_event("startup")
    async def startup():
        os.makedirs(os.path.dirname(settings.sqlite_db_path), exist_ok=True)
        await init_db()
        logger.info("SafeSpace AI started. Webhook: /whatsapp/webhook")

    return combined


def _loading_html() -> str:
    """Simple loading page shown while Streamlit is starting up."""
    return """<!DOCTYPE html>
<html>
<head>
    <title>SafeSpace AI — Starting...</title>
    <meta http-equiv="refresh" content="3">
    <style>
        body { font-family: sans-serif; display: flex; align-items: center;
               justify-content: center; height: 100vh; margin: 0;
               background: #0e1117; color: #fafafa; flex-direction: column; }
        .logo { font-size: 3rem; margin-bottom: 1rem; }
        h1 { font-size: 1.5rem; color: #4CAF50; }
        p { color: #aaa; }
    </style>
</head>
<body>
    <div class="logo">🌿</div>
    <h1>SafeSpace AI</h1>
    <p>Starting up... page will refresh automatically.</p>
</body>
</html>"""


# ── Main entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start Streamlit in background
    streamlit_proc = start_streamlit()

    # Wait a moment for Streamlit to start
    logger.info("Waiting for Streamlit to start...")
    time.sleep(8)

    # Create and run the combined FastAPI app on port 7860
    import uvicorn
    app = create_combined_app()
    logger.info("Starting FastAPI proxy on port 7860 (HF public port)...")
    logger.info("WhatsApp webhook URL: https://hiteshiaglawe0505-safespace-ai.hf.space/whatsapp/webhook")

    try:
        uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
    finally:
        # Clean up Streamlit process if FastAPI stops
        streamlit_proc.terminate()