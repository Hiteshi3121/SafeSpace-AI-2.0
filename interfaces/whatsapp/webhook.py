"""
interfaces/whatsapp/webhook.py

WhatsApp channel adapter.
Translates Twilio webhook payloads → ChatRequest,
then formats ChatResponse → WhatsApp message.

This is the ONLY file that knows about Twilio/WhatsApp.
The core engine sees nothing but a ChatRequest.
"""

import logging
import httpx
from fastapi import APIRouter, Form, BackgroundTasks
from fastapi.responses import Response

from core.schemas import ChatRequest, ChatResponse, MessageType
from core.engine import handle_request
from interfaces.whatsapp.sender import send_whatsapp_message

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/webhook")
async def whatsapp_webhook(
    background_tasks: BackgroundTasks,
    From: str = Form(...),
    Body: str = Form(""),
    MediaContentType0: str | None = Form(None),
    MediaUrl0: str | None = Form(None),
    NumMedia: int = Form(0),
):
    """
    Receives Twilio WhatsApp webhook.
    Detects message type (text / audio / image) and dispatches to core engine.
    Returns empty 200 immediately — response is sent asynchronously.
    """
    user_id = From  # e.g. "whatsapp:+919876543210"
    logger.info("Incoming WhatsApp message from %s | media=%d", user_id, NumMedia)

    # Build ChatRequest based on message type
    request = await _build_request(
        user_id=user_id,
        body=Body,
        media_url=MediaUrl0,
        media_content_type=MediaContentType0,
        num_media=NumMedia,
    )

    # Process in background so Twilio doesn't timeout (15s limit)
    background_tasks.add_task(_process_and_reply, request)

    # Acknowledge receipt immediately
    return Response(content="", media_type="text/xml", status_code=200)


async def _process_and_reply(request: ChatRequest) -> None:
    """Runs the core engine and sends the response back to WhatsApp."""
    from core.config import get_settings
    settings = get_settings()

    try:
        response: ChatResponse = await handle_request(request)
        reply_text = _format_response(response)

        # If Twilio not configured (dev/test mode), just log the response
        if not settings.twilio_configured:
            logger.info(
                "TWILIO NOT CONFIGURED — response for %s:\n%s",
                request.user_id, reply_text
            )
            return

        await send_whatsapp_message(to=request.user_id, body=reply_text)

    except Exception as e:
        logger.exception("Failed to process and reply for %s: %s", request.user_id, e)
        if not settings.twilio_configured:
            return  # Can't send error message either — just log and exit cleanly
        try:
            await send_whatsapp_message(
                to=request.user_id,
                body="I'm sorry, something went wrong. Please try again in a moment. 🙏",
            )
        except Exception:
            logger.error("Also failed to send error message to %s", request.user_id)


async def _build_request(
    user_id: str,
    body: str,
    media_url: str | None,
    media_content_type: str | None,
    num_media: int,
) -> ChatRequest:
    """
    Determines message type and fetches media bytes if needed.
    """
    if num_media == 0 or not media_url:
        # Plain text message
        return ChatRequest(
            user_id=user_id,
            message_type=MessageType.TEXT,
            text=body.strip() or "Hello",
        )

    # Download media from Twilio
    media_bytes = await _download_media(media_url)

    if media_content_type and "audio" in media_content_type:
        return ChatRequest(
            user_id=user_id,
            message_type=MessageType.AUDIO,
            audio_bytes=media_bytes,
            text=body.strip() if body.strip() else None,
            metadata={"content_type": media_content_type},
        )

    if media_content_type and "image" in media_content_type:
        return ChatRequest(
            user_id=user_id,
            message_type=MessageType.IMAGE,
            image_bytes=media_bytes,
            text=body.strip() if body.strip() else None,  # Caption as context
            metadata={"content_type": media_content_type},
        )

    # Fallback — treat body as text
    return ChatRequest(
        user_id=user_id,
        message_type=MessageType.TEXT,
        text=body.strip() or "Hello",
    )


async def _download_media(media_url: str) -> bytes:
    """Download media file from Twilio CDN."""
    from core.config import get_settings
    settings = get_settings()

    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(
            media_url,
            auth=(settings.twilio_account_sid, settings.twilio_auth_token),
            timeout=30.0,
        )
        response.raise_for_status()
        return response.content


def _format_response(response: ChatResponse) -> str:
    """
    Format ChatResponse for WhatsApp.
    WhatsApp is plain text only — no markdown rendering.
    """
    text = response.text

    if response.escalated:
        text += "\n\n🚨 Emergency services have been notified."

    if response.has_error:
        logger.warning("Response contains error: %s", response.error)

    return text