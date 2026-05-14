"""
hf_app.py — SafeSpace AI combined server

ARCHITECTURE (with Nginx):
===========================
Nginx (port 7860, public)
  ├── /whatsapp/*  →  FastAPI (port 8000) — Twilio webhook
  ├── /health      →  FastAPI (port 8000) — health check
  └── /*           →  Streamlit (port 8501) — Web UI with proper WebSocket

This file starts all 3 processes:
  1. FastAPI on 8000
  2. Streamlit on 8501
  3. Nginx on 7860

WHY NGINX instead of Python proxy:
  Nginx properly upgrades HTTP→WebSocket (Upgrade header).
  Python's httpx-based proxy blocks WebSocket, causing the 403 error
  and skeleton loading screen you saw before.

TWILIO WEBHOOK URL:
  https://hiteshiaglawe0505-safespace-ai.hf.space/whatsapp/webhook
"""

import sys
import os
import subprocess
import threading
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)

ROOT = Path("/app")
sys.path.insert(0, str(ROOT))
os.makedirs(str(ROOT / "data"), exist_ok=True)

# ── Inject secrets ─────────────────────────────────────────────────────────────
try:
    from core.config import get_settings
    _s = get_settings()
    for k, v in {
        "GROQ_API_KEY":         _s.groq_api_key,
        "GOOGLE_MAPS_API_KEY":  _s.google_maps_api_key,
        "LANGCHAIN_API_KEY":    _s.langsmith_api_key,
        "LANGSMITH_API_KEY":    _s.langsmith_api_key,
        "LANGCHAIN_TRACING_V2": "true" if _s.langsmith_api_key else "false",
        "LANGCHAIN_PROJECT":    _s.langsmith_project,
    }.items():
        if v:
            os.environ.setdefault(k, v)
    logger.info("Secrets loaded — Twilio=%s Maps=%s LangSmith=%s",
                _s.twilio_configured, _s.maps_configured, _s.langsmith_configured)
except Exception as e:
    logger.warning("Settings warning: %s", e)


def start_fastapi():
    """FastAPI on port 8000 — handles WhatsApp webhook and health check."""
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from core.config import get_settings
    from memory.store import init_db
    from interfaces.whatsapp.webhook import router as whatsapp_router

    settings = get_settings()
    app = FastAPI(title="SafeSpace Webhook")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(whatsapp_router, prefix="/whatsapp")

    @app.get("/health")
    async def health():
        return {
            "status": "ok", "version": "2.0.0",
            "twilio": settings.twilio_configured,
            "maps":   settings.maps_configured,
        }

    @app.on_event("startup")
    async def startup():
        os.makedirs(os.path.dirname(settings.sqlite_db_path), exist_ok=True)
        await init_db()
        logger.info("FastAPI ready — webhook at /whatsapp/webhook")

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


def start_streamlit():
    """Streamlit on port 8501 — the Web UI."""
    app_path = str(ROOT / "interfaces" / "streamlit_ui" / "app.py")
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ]
    logger.info("Starting Streamlit on port 8501...")
    proc = subprocess.Popen(cmd, env=os.environ.copy())
    return proc


def start_nginx():
    """Nginx on port 7860 — routes traffic to FastAPI and Streamlit."""
    # Update nginx pid file path to writable location
    nginx_conf = """
worker_processes 1;
pid /tmp/nginx/nginx.pid;
error_log /tmp/nginx/error.log;

events { worker_connections 1024; }

http {
    access_log /tmp/nginx/access.log;
    
    server {
        listen 7860;
        
        location /whatsapp/ {
            proxy_pass http://127.0.0.1:8000/whatsapp/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_read_timeout 60s;
        }
        
        location /health {
            proxy_pass http://127.0.0.1:8000/health;
            proxy_set_header Host $host;
        }
        
        location / {
            proxy_pass http://127.0.0.1:8501/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_read_timeout 86400s;
        }
    }
}
"""
    os.makedirs("/tmp/nginx", exist_ok=True)
    conf_path = "/tmp/nginx/nginx.conf"
    with open(conf_path, "w") as f:
        f.write(nginx_conf)

    logger.info("Starting Nginx on port 7860...")
    cmd = ["nginx", "-c", conf_path, "-g", "daemon off;"]
    proc = subprocess.Popen(cmd)
    return proc


if __name__ == "__main__":
    logger.info("="*55)
    logger.info("SafeSpace AI 2.0 — Starting all services")
    logger.info("  Streamlit (Web UI):  port 8501 (internal)")
    logger.info("  FastAPI  (Webhook):  port 8000 (internal)")
    logger.info("  Nginx    (Public):   port 7860 → routes both")
    logger.info("="*55)

    # 1. Start FastAPI in background thread
    fastapi_thread = threading.Thread(target=start_fastapi, daemon=True, name="FastAPI")
    fastapi_thread.start()

    # 2. Start Streamlit subprocess
    streamlit_proc = start_streamlit()

    # 3. Wait for both to be ready before starting Nginx
    logger.info("Waiting 10s for services to initialize...")
    time.sleep(10)

    # 4. Start Nginx (routes all public traffic)
    nginx_proc = start_nginx()

    logger.info("All services running. Webhook URL:")
    logger.info("  https://hiteshiaglawe0505-safespace-ai.hf.space/whatsapp/webhook")

    try:
        # Wait for Nginx (our main process)
        nginx_proc.wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        streamlit_proc.terminate()
        nginx_proc.terminate()