"""
core/schemas.py

The single contract between every UI channel and the core engine.

Any channel (WhatsApp, Streamlit, Telegram, REST API) must translate
its input into ChatRequest and will receive a ChatResponse back.
The core engine never knows which channel is calling it.
"""

from enum import Enum
from pydantic import BaseModel, Field


class Intent(str, Enum):
    MEDICAL = "MEDICAL"
    THERAPY = "THERAPY"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class MessageType(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"


class ChatRequest(BaseModel):
    """
    Normalised input from any channel.
    Exactly one of text / audio_bytes / image_bytes must be set.
    """
    user_id: str = Field(..., description="Unique identifier for the user/session")
    message_type: MessageType = MessageType.TEXT
    text: str | None = Field(None, description="Raw text from user")
    audio_bytes: bytes | None = Field(None, description="Raw audio bytes (OGG from WhatsApp)")
    image_bytes: bytes | None = Field(None, description="Raw image bytes")
    metadata: dict = Field(default_factory=dict, description="Channel-specific extras")


class ChatResponse(BaseModel):
    """
    Normalised output returned to any channel.
    The channel formats this however its UI needs.
    """
    text: str = Field(..., description="Response text to send to the user")
    intent: Intent = Field(Intent.UNKNOWN, description="Classified intent of the user message")
    escalated: bool = Field(False, description="True if emergency call was triggered")
    therapist_results: list[dict] = Field(
        default_factory=list,
        description="Nearby therapist results if location lookup was triggered"
    )
    error: str | None = Field(None, description="Set if something went wrong gracefully")

    @property
    def has_error(self) -> bool:
        return self.error is not None


class SessionMessage(BaseModel):
    """A single message in a user's conversation history."""
    role: str  # "user" or "assistant"
    content: str


class UserSession(BaseModel):
    """Full session state for a user stored in SQLite."""
    user_id: str
    messages: list[SessionMessage] = Field(default_factory=list)
    message_count: int = 0