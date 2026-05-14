"""
multimodal/vision.py
Medical image analysis using Llama 4 Scout (vision) on Groq.
"""

import base64
import logging
from groq import AsyncGroq
from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_client: AsyncGroq | None = None

VISION_SYSTEM_PROMPT = """You are a medical image analysis assistant for SafeSpace AI.

When given an image, provide:
1. What you clearly observe (rash, wound, skin condition, medical report, etc.)
2. Visual details a doctor would note (colour, pattern, distribution, severity)
3. Any visible text if it's a medical report or prescription

RULES:
- Never diagnose. Use "this appears to be" or "this resembles"
- Always recommend consulting a real doctor
- Keep descriptions factual and clinical, not alarming
- Be concise — 3-5 sentences maximum
"""


def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=settings.groq_api_key)
    return _client


async def describe_image(image_bytes: bytes, media_type: str = "image/jpeg") -> str:
    """
    Analyse a medical image and return a text description.
    Tries multiple Groq vision models in order of preference.
    """
    # Try models in order — Groq updates model names frequently
    models_to_try = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-90b-vision-preview",
    ]

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    client = _get_client()

    for model in models_to_try:
        try:
            logger.info("Trying vision model: %s", model)
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": VISION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{b64_image}"
                                },
                            },
                            {
                                "type": "text",
                                "text": "Please analyse this medical image and describe what you observe.",
                            },
                        ],
                    },
                ],
                max_tokens=400,
            )
            description = response.choices[0].message.content.strip()
            logger.info("Vision analysis complete with model %s (%d chars)", model, len(description))
            return f"[Image analysis] {description}"

        except Exception as e:
            logger.warning("Vision model %s failed: %s — trying next", model, e)
            continue

    logger.error("All vision models failed")
    return (
        "I received your image but couldn't analyse it right now. "
        "Could you describe what you're seeing in text?"
    )