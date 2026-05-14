"""
core/config.py

Single source of truth for all environment variables.
Uses pydantic-settings — validates everything at startup.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────
    groq_api_key: str

    # ── LLMOps ───────────────────────────────────────────────────
    langsmith_api_key: str = ""
    langsmith_project: str = "safespace-ai"
    langsmith_tracing: bool = True

    # ── Twilio ───────────────────────────────────────────────────
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_number: str = "whatsapp:+14155238886"
    twilio_from_number: str = ""
    emergency_contact: str = ""

    # ── Google Maps ───────────────────────────────────────────────
    # Required for therapist finder feature.
    # Get key: console.cloud.google.com → APIs → Enable "Places API"
    google_maps_api_key: str = ""

    # ── MCP Server (optional) ─────────────────────────────────────
    # When set, maps_tool.py routes therapist searches through the MCP server.
    # Leave empty to call Google Maps directly (recommended for local dev).
    # Local:    THERAPIST_MCP_URL=http://localhost:8001
    # Deployed: THERAPIST_MCP_URL=https://your-mcp-server.com
    therapist_mcp_url: str = ""

    # ── App ───────────────────────────────────────────────────────
    app_env: str = "development"
    app_port: int = 8000
    log_level: str = "INFO"

    # ── Memory ────────────────────────────────────────────────────
    sqlite_db_path: str = "./data/safespace.db"
    session_max_messages: int = 20

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def twilio_configured(self) -> bool:
        return bool(self.twilio_account_sid and self.twilio_auth_token)

    @property
    def maps_configured(self) -> bool:
        return bool(self.google_maps_api_key)

    @property
    def langsmith_configured(self) -> bool:
        return bool(self.langsmith_api_key and self.langsmith_tracing)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()