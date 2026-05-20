"""
core/engine.py
The core engine — the only thing every channel adapter calls.
"""

import logging
from core.schemas import ChatRequest, ChatResponse, Intent, MessageType
from core.config import get_settings
from memory.store import get_session, save_message
from observability.tracer import trace_request, log_intent

logger = logging.getLogger(__name__)
settings = get_settings()


async def handle_request(request: ChatRequest) -> ChatResponse:
    """
    Central request handler.

    HOW TRACING WORKS:
    trace_request() returns a @traceable-decorated function.
    When run_crew() is called inside it, CrewAI/LiteLLM's callback
    handlers see the active RunTree context and automatically nest
    all LLM calls under this parent → LLM Calls tab populates in LangSmith.
    """
    # Detect channel from user_id for metadata
    channel = "whatsapp" if "whatsapp" in request.user_id else "web"

    try:
        user_text = await _resolve_text(request)
        if not user_text:
            return ChatResponse(
                text="I didn't receive any message. Please try again.",
                error="empty_input",
            )

        session = await get_session(request.user_id)

        from agents.crew import run_crew

        # Get @traceable wrapper — creates parent span in LangSmith
        tracer = trace_request(
            user_id=request.user_id,
            channel=channel,
            message_type=request.message_type.value if request.message_type else "text",
        )

        if tracer:
            # Run inside @traceable context — all CrewAI LLM calls nest under it
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: tracer(
                    lambda: asyncio.run(run_crew(
                        user_text=user_text,
                        session=session,
                        user_id=request.user_id,
                    ))
                )
            )
        else:
            # No tracing — run directly
            result = await run_crew(
                user_text=user_text,
                session=session,
                user_id=request.user_id,
            )

        await save_message(request.user_id, role="user", content=user_text)
        await save_message(request.user_id, role="assistant", content=result.text)

        return result

    except Exception as e:
        logger.exception("Unhandled error in core engine for user %s", request.user_id)
        return ChatResponse(
            text="I'm sorry, something went wrong. Please try again in a moment.",
            error=str(e),
        )


async def _resolve_text(request: ChatRequest) -> str | None:
    """
    Converts multimodal input to plain text.

    KEY FIX: When message_type is IMAGE, ALWAYS run vision analysis
    even if there's also a text caption. The caption becomes context
    for the vision model ("please diagnose my face" + image bytes).

    Priority:
      IMAGE type → vision analysis (+ caption as context if present)
      AUDIO type → Whisper transcription
      TEXT type  → plain text
    """
    from multimodal.speech import transcribe_audio
    from multimodal.vision import describe_image

    # Image — always analyze, include caption as context
    if request.message_type == MessageType.IMAGE and request.image_bytes:
        logger.info("Running vision analysis for user %s", request.user_id)
        media_type = request.metadata.get("image_media_type", "image/jpeg")
        vision_result = await describe_image(request.image_bytes, media_type=media_type)
        # If user also sent a caption, include it
        if request.text and request.text.strip():
            return f"User's message: '{request.text.strip()}'\n\n{vision_result}"
        return vision_result

    # Audio — transcribe
    if request.message_type == MessageType.AUDIO and request.audio_bytes:
        logger.info("Transcribing audio for user %s", request.user_id)
        # Use text field as filename hint if provided by channel
        filename = request.text if request.text and "." in str(request.text) else "audio.ogg"
        # Clear text so filename hint is not treated as user message
        return await transcribe_audio(request.audio_bytes, filename=filename)

    # Text — plain text
    if request.text:
        return request.text.strip()

    # Fallback — try any available content
    if request.audio_bytes:
        return await transcribe_audio(request.audio_bytes)
    if request.image_bytes:
        return await describe_image(request.image_bytes)

    return None



# """
# core/engine.py
# The core engine — the only thing every channel adapter calls.
# """

# import logging
# from core.schemas import ChatRequest, ChatResponse, Intent, MessageType
# from core.config import get_settings
# from memory.store import get_session, save_message
# from observability.tracer import trace_request

# logger = logging.getLogger(__name__)
# settings = get_settings()


# async def handle_request(request: ChatRequest) -> ChatResponse:
#     with trace_request(user_id=request.user_id):
#         try:
#             user_text = await _resolve_text(request)
#             if not user_text:
#                 return ChatResponse(
#                     text="I didn't receive any message. Please try again.",
#                     error="empty_input",
#                 )

#             session = await get_session(request.user_id)

#             from agents.crew import run_crew
#             result = await run_crew(
#                 user_text=user_text,
#                 session=session,
#                 user_id=request.user_id,
#             )

#             await save_message(request.user_id, role="user", content=user_text)
#             await save_message(request.user_id, role="assistant", content=result.text)

#             return result

#         except Exception as e:
#             logger.exception("Unhandled error in core engine for user %s", request.user_id)
#             return ChatResponse(
#                 text="I'm sorry, something went wrong. Please try again in a moment.",
#                 error=str(e),
#             )


# async def _resolve_text(request: ChatRequest) -> str | None:
#     """
#     Converts multimodal input to plain text.

#     KEY FIX: When message_type is IMAGE, ALWAYS run vision analysis
#     even if there's also a text caption. The caption becomes context
#     for the vision model ("please diagnose my face" + image bytes).

#     Priority:
#       IMAGE type → vision analysis (+ caption as context if present)
#       AUDIO type → Whisper transcription
#       TEXT type  → plain text
#     """
#     from multimodal.speech import transcribe_audio
#     from multimodal.vision import describe_image

#     # Image — always analyze, include caption as context
#     if request.message_type == MessageType.IMAGE and request.image_bytes:
#         logger.info("Running vision analysis for user %s", request.user_id)
#         media_type = request.metadata.get("image_media_type", "image/jpeg")
#         vision_result = await describe_image(request.image_bytes, media_type=media_type)
#         # If user also sent a caption, include it
#         if request.text and request.text.strip():
#             return f"User's message: '{request.text.strip()}'\n\n{vision_result}"
#         return vision_result

#     # Audio — transcribe
#     if request.message_type == MessageType.AUDIO and request.audio_bytes:
#         logger.info("Transcribing audio for user %s", request.user_id)
#         # Use text field as filename hint if provided by channel
#         filename = request.text if request.text and "." in str(request.text) else "audio.ogg"
#         # Clear text so filename hint is not treated as user message
#         return await transcribe_audio(request.audio_bytes, filename=filename)

#     # Text — plain text
#     if request.text:
#         return request.text.strip()

#     # Fallback — try any available content
#     if request.audio_bytes:
#         return await transcribe_audio(request.audio_bytes)
#     if request.image_bytes:
#         return await describe_image(request.image_bytes)

#     return None