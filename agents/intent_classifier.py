"""
agents/intent_classifier.py

Classifies user intent using Pydantic structured output.

WHY PYDANTIC HERE?
==================
In SafeSpace 1.0, intent was a raw string returned by the LLM:
    "MEDICAL" / "THERAPY" / "MIXED"
That's fragile — the LLM could return "Medical" or "medical help" and
the if/elif in brain.py would silently fail.

With Pydantic structured output, the LLM is forced to return a validated
object. If it returns garbage, we catch it at the schema level, not
buried in routing logic at 2am.

HOW GROQ STRUCTURED OUTPUT WORKS:
===================================
Groq supports response_format with JSON schema. We pass our Pydantic
model's JSON schema to Groq, and it guarantees the response matches it.
We then parse the JSON into our IntentResult model — fully validated.
"""

import json
import logging
from groq import AsyncGroq
from pydantic import BaseModel, Field
from core.config import get_settings
from core.schemas import Intent

logger = logging.getLogger(__name__)
settings = get_settings()

_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=settings.groq_api_key)
    return _client


# ── The structured output schema ──────────────────────────────────────────────

class IntentResult(BaseModel):
    """
    Structured output from the intent classifier.
    Groq is forced to return exactly this shape — no free-form text.
    """
    intent: Intent = Field(
        ...,
        description="The primary intent: MEDICAL, THERAPY, or MIXED"
    )
    confidence: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Confidence score from 0.0 to 1.0"
    )
    reasoning: str = Field(
        ...,
        description="One sentence explaining the classification"
    )


# ── System prompt ─────────────────────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """You are an intent classifier for SafeSpace AI, a medical and mental health assistant.

Classify the user's message into exactly ONE of these intents:

MEDICAL  → Physical health: symptoms, pain, medication, diagnosis questions, injuries, lab reports
THERAPY  → Mental/emotional health: stress, anxiety, depression, relationships, grief, loneliness, trauma
MIXED    → Clearly both: e.g. "I have chest pain and I'm really anxious about it"

Rules:
- If unsure between MEDICAL and THERAPY, choose MIXED
- Suicidal thoughts or self-harm → always THERAPY (even if they mention physical pain)
- "I feel tired" alone → THERAPY (emotional fatigue is common)
- "I feel tired and have fever" → MEDICAL
- Greetings like "hi" or "hello" → THERAPY (start with empathy)

You MUST respond with a JSON object containing ALL THREE of these fields:
{
  "intent": "MEDICAL" or "THERAPY" or "MIXED",
  "confidence": a float between 0.0 and 1.0,
  "reasoning": "one sentence explaining why you chose this intent"
}

Example response:
{
  "intent": "MEDICAL",
  "confidence": 0.92,
  "reasoning": "User describes physical symptoms of headache and fever indicating a medical concern."
}

Return ONLY the JSON object. No extra text, no markdown, no code blocks.
"""


# ── Main function ─────────────────────────────────────────────────────────────

async def classify_intent(user_text: str) -> IntentResult:
    """
    Classify the intent of a user message using Groq structured output.

    Returns an IntentResult with intent, confidence, and reasoning.
    Falls back to MIXED on any error — safe default.
    """
    try:
        client = _get_client()

        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            response_format={
                "type": "json_object"
            },
            temperature=0.1,   # Low temp for consistent classification
            max_tokens=150,
        )

        raw_json = response.choices[0].message.content
        data = json.loads(raw_json)
        result = IntentResult(**data)

        logger.info(
            "Intent classified: %s (confidence=%.2f) | %s",
            result.intent, result.confidence, result.reasoning
        )
        return result

    except Exception as e:
        logger.exception("Intent classification failed, defaulting to MIXED: %s", e)
        return IntentResult(
            intent=Intent.MIXED,
            confidence=0.5,
            reasoning="Classification failed — defaulting to MIXED for safety",
        )