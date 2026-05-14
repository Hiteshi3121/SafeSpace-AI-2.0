"""
multimodal/speech.py
Audio transcription using Whisper large-v3 on Groq.
Accepts raw bytes in any common audio format.
"""

import io
import logging
from groq import AsyncGroq
from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=settings.groq_api_key)
    return _client


def _detect_format(audio_bytes: bytes, hint_filename: str = "audio.ogg") -> tuple[str, str]:
    """
    Detect audio format from magic bytes or filename hint.
    Returns (filename, mime_type) tuple.
    """
    # Check magic bytes
    if audio_bytes[:4] == b'OggS':
        return "audio.ogg", "audio/ogg"
    if audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
        return "audio.mp3", "audio/mpeg"
    if audio_bytes[:4] == b'RIFF':
        return "audio.wav", "audio/wav"
    if audio_bytes[:4] == b'ftyp' or audio_bytes[4:8] == b'ftyp':
        return "audio.m4a", "audio/mp4"
    if audio_bytes[:4] == b'\x1aE\xdf\xa3':
        return "audio.webm", "audio/webm"

    # Fall back to hint filename extension
    ext = hint_filename.rsplit('.', 1)[-1].lower() if '.' in hint_filename else 'ogg'
    mime_map = {
        'ogg': 'audio/ogg', 'mp3': 'audio/mpeg', 'wav': 'audio/wav',
        'm4a': 'audio/mp4', 'webm': 'audio/webm', 'flac': 'audio/flac'
    }
    mime = mime_map.get(ext, 'audio/ogg')
    return f"audio.{ext}", mime


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.ogg") -> str:
    """
    Transcribe raw audio bytes to text using Whisper large-v3 on Groq.
    Auto-detects audio format from magic bytes.
    """
    if not audio_bytes:
        return "I couldn't hear anything. Please try again or send a text message."

    try:
        client = _get_client()
        detected_filename, mime_type = _detect_format(audio_bytes, filename)

        logger.info("Transcribing audio: %s (%d bytes)", detected_filename, len(audio_bytes))

        audio_file = io.BytesIO(audio_bytes)

        transcription = await client.audio.transcriptions.create(
            file=(detected_filename, audio_file),
            model="whisper-large-v3",
            response_format="text",
            language="en",
        )

        text = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()

        if not text:
            return "I had trouble understanding the audio. Could you please repeat or type your message?"

        logger.info("Transcription complete (%d chars): %s...", len(text), text[:50])
        return text

    except Exception as e:
        err_str = str(e)
        logger.exception("Transcription failed: %s", err_str)
        if "rate_limit" in err_str.lower() or "429" in err_str:
            return (
                "⚠️ Voice transcription is rate-limited right now. "
                "Please wait a minute and try again, or type your message instead."
            )
        elif "invalid_api_key" in err_str.lower() or "401" in err_str:
            return "⚠️ API key issue with voice transcription. Check GROQ_API_KEY in Space settings."
        return (
            "⚠️ Voice transcription failed. Please type your message instead. "
            "You can try again in a few minutes."
        )