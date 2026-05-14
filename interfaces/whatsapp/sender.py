"""
interfaces/whatsapp/sender.py

Sends WhatsApp messages via Twilio.
Lazy singleton — Twilio client only initialised if configured.
"""

import logging
from twilio.rest import Client
from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_twilio_client: Client | None = None


def _get_twilio_client() -> Client:
    global _twilio_client
    if _twilio_client is None:
        if not settings.twilio_configured:
            raise RuntimeError(
                "Twilio not configured. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in .env"
            )
        _twilio_client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    return _twilio_client


async def send_whatsapp_message(to: str, body: str) -> None:
    """
    Send a WhatsApp message via Twilio.

    Args:
        to: Recipient number in WhatsApp format e.g. "whatsapp:+919876543210"
        body: Plain text message body
    """
    try:
        client = _get_twilio_client()

        # Ensure the 'to' number is in whatsapp: format
        if not to.startswith("whatsapp:"):
            to = f"whatsapp:{to}"

        # Split long messages (WhatsApp limit is 1600 chars)
        chunks = _split_message(body, max_length=1500)
        for chunk in chunks:
            client.messages.create(
                from_=settings.twilio_whatsapp_number,
                to=to,
                body=chunk,
            )
            logger.info("WhatsApp message sent to %s (%d chars)", to, len(chunk))

    except RuntimeError:
        raise
    except Exception as e:
        logger.exception("Failed to send WhatsApp message to %s: %s", to, e)
        raise


def _split_message(text: str, max_length: int = 1500) -> list[str]:
    """Split long messages into chunks that fit WhatsApp's limit."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        # Split at last newline before max_length
        split_at = text.rfind("\n", 0, max_length)
        if split_at == -1:
            split_at = max_length
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()

    return chunks