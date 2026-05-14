"""
app.py

Thin FastAPI application shell.
Registers all routers — does zero business logic itself.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
from memory.store import init_db

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("SafeSpace AI starting up (env=%s)", settings.app_env)
    os.makedirs(os.path.dirname(settings.sqlite_db_path), exist_ok=True)
    await init_db()
    logger.info("SQLite memory initialised at %s", settings.sqlite_db_path)
    yield
    logger.info("SafeSpace AI shutting down")


app = FastAPI(
    title="SafeSpace AI",
    description="AI Medical & Mental Health Assistant",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ───────────────────────────────────────────────────────────────────
from interfaces.whatsapp.webhook import router as whatsapp_router  # noqa: E402

app.include_router(whatsapp_router, prefix="/whatsapp", tags=["WhatsApp"])


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "env": settings.app_env,
        "twilio": settings.twilio_configured,
        "maps": settings.maps_configured,
        "langsmith": settings.langsmith_configured,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=settings.is_development,
    )