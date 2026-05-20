"""
agents/crew.py
The CrewAI crew — wires all agents + tasks together.
"""

import asyncio
import logging
import re
from crewai import Crew, Task, Process

from agents.intent_classifier import classify_intent
from agents.safety import create_safety_agent
from agents.doctor import create_doctor_agent
from agents.therapist import create_therapist_agent
from core.schemas import ChatResponse, Intent, UserSession
from memory.store import format_history_for_llm

logger = logging.getLogger(__name__)


def _clean_crew_response(text: str) -> str:
    """Remove LiteLLM/CrewAI artifacts from agent responses."""
    # Remove math boxed format leak: $\boxed{...}$
    text = re.sub(r'Your response should be in this format:.*?\\boxed\{[^}]*\}\.?\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'The final answer is:\s*\$\\boxed\{', '', text)
    text = re.sub(r'\$\\boxed\{(.*?)\}\$', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\\boxed\{(.*?)\}', r'\1', text, flags=re.DOTALL)
    # Remove tool call leakage
    text = re.sub(r'<function=\w+>.*?(?:</function>|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'\{"location":\s*"[^"]*".*?\}', '', text, flags=re.DOTALL)
    # Remove "The final answer is:" preamble
    text = re.sub(r'^The final answer is:\s*', '', text.strip())
    return text.strip()


# ── Greeting detection ────────────────────────────────────────────────────────
GREETINGS = {
    "hi", "hello", "hey", "hii", "helo", "heya", "howdy",
    "good morning", "good afternoon", "good evening", "good night",
    "namaste", "namaskar", "sup", "what's up", "whats up", "yo"
}

GREETING_RESPONSES = [
    "Hello! 🌿 I'm SafeSpace, your AI health companion. I'm here to help with medical questions or emotional support. What's on your mind today?",
    "Hi there! 🌿 Welcome to SafeSpace. Whether it's a health concern or just need someone to talk to — I'm here. How can I help you today?",
    "Hey! 🌿 I'm SafeSpace — your personal medical and mental health assistant. What would you like to talk about today?",
    "Hello! 🌿 Great to hear from you. I can help with medical questions, emotional support, or finding a therapist near you. What do you need today?",
]


def _is_greeting(text: str) -> bool:
    cleaned = text.lower().strip().rstrip("!?.").strip()
    return cleaned in GREETINGS or len(cleaned) < 4


def _get_greeting_response() -> str:
    import random
    return random.choice(GREETING_RESPONSES)


def _trim_response(text: str, max_chars: int = 1200) -> str:
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars]
    last_period = max(trimmed.rfind('.'), trimmed.rfind('!'), trimmed.rfind('?'))
    if last_period > max_chars * 0.7:
        return trimmed[:last_period + 1]
    return trimmed + "..."



def _detect_location_followup(user_text: str, session) -> str | None:
    """
    Detects when user replies with a location after AI asked for their city.
    Uses LLM to extract the city name from any phrasing.

    Examples it handles:
      "yes, nagpur"
      "nagpur"
      "yes tell me therapist in wadi nagpur"  → extracts "wadi, nagpur"
      "moosapet hyderabad"
      "I am in Banjara Hills, Hyderabad"      → extracts "Banjara Hills, Hyderabad"
    """
    if not session or not session.messages:
        return None

    # Find last AI message
    last_ai = next(
        (m.content for m in reversed(session.messages) if m.role == "assistant"),
        None
    )
    if not last_ai:
        return None

    # Did the AI ask for a city in any phrasing?
    # NOTE: must match whatever the TherapistAgent actually outputs —
    # the agent often varies the phrasing so we cast a wide net here.
    asked_for_city = any(phrase in last_ai.lower() for phrase in [
        "reply with your city",
        "reply with your city name",
        "tell me your city",
        "what city are you in",
        "your city name",
        "city name",
        "find a therapist near you",   # catches "Would you like me to find a therapist near you?"
        "therapist near you",          # catches varied phrasings
        "find you a therapist",
        "nearest therapist",
        "therapist in your area",
        "therapist near",
        "find therapist",
    ])
    if not asked_for_city:
        return None

    # User said no — don't search
    user_lower = user_text.lower().strip()
    if user_lower in ["no", "nope", "na", "nahi", "not now", "no thanks", "skip"]:
        return "No problem! Let me know if you need anything else. 🌿"

    # Use LLM to extract the city/location from whatever the user typed
    # This handles: "yes nagpur", "wadi nagpur", "I am in Banjara Hills Hyderabad", etc.
    location = _extract_location_with_llm(user_text)

    if not location:
        return None

    # Call therapist finder directly
    from tools.maps_tool import find_therapists_tool
    logger.info("Location follow-up: searching therapists near '%s'", location)
    return find_therapists_tool._run(location=location)


def _extract_location_with_llm(user_text: str) -> str | None:
    """
    Uses LLM to extract just the location/city name from user message.
    Returns the location string, or None if no location found.
    """
    from groq import Groq
    from core.config import get_settings
    settings = get_settings()

    try:
        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract only the city or location name from the user message. "
                        "Reply with ONLY the location name — nothing else. "
                        "Examples: "
                        "'yes nagpur' → 'Nagpur' | "
                        "'find therapist in wadi nagpur' → 'Wadi, Nagpur' | "
                        "'I am in Banjara Hills Hyderabad' → 'Banjara Hills, Hyderabad' | "
                        "'moosapet hyderabad' → 'Moosapet, Hyderabad' | "
                        "'no' → 'NONE' | "
                        "'I don't know' → 'NONE'. "
                        "If no location present, reply NONE."
                    )
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ],
            max_tokens=15,
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip()
        logger.info("LLM extracted location: '%s' from '%s'", result, user_text)

        if result.upper() == "NONE" or not result:
            return None
        return result

    except Exception as e:
        logger.warning("LLM location extraction failed: %s", e)
        # Fallback: simple strip of common prefixes
        loc = user_text.lower().strip()
        for prefix in ["yes,", "yes ", "sure,", "sure ", "ok ", "find ", "search "]:
            if loc.startswith(prefix):
                loc = loc[len(prefix):].strip()
        return loc.title() if len(loc) > 2 else None



async def run_crew(
    user_text: str,
    session: UserSession,
    user_id: str,
) -> ChatResponse:
    """Main crew orchestration function."""

    # ── Fast path: greeting detection ─────────────────────────────────────────
    if _is_greeting(user_text):
        logger.info("User %s | Greeting detected — fast path response", user_id)
        return ChatResponse(
            text=_get_greeting_response(),
            intent=Intent.UNKNOWN,
            escalated=False,
        )

    # ── Step 1: Classify intent ────────────────────────────────────────────────
    # ── Location follow-up: bypass crew if user replied with a city ──────────
    location_response = _detect_location_followup(user_text, session)
    if location_response:
        return ChatResponse(
            text=location_response,
            intent=Intent.THERAPY,
            escalated=False,
        )

    intent_result = await classify_intent(user_text)
    logger.info("User %s | Intent: %s (%.2f)", user_id, intent_result.intent, intent_result.confidence)

    # Log to LangSmith
    from observability.tracer import log_intent
    log_intent(user_id=user_id, intent=intent_result.intent.value, confidence=intent_result.confidence)

    # ── Step 2: Format conversation history ───────────────────────────────────
    history = format_history_for_llm(session)

    # ── Step 3: Create agents ─────────────────────────────────────────────────
    safety_agent = create_safety_agent()
    doctor_agent = create_doctor_agent()
    therapist_agent = create_therapist_agent()

    # ── Step 4: Build tasks ────────────────────────────────────────────────────
    safety_task = Task(
        description=f"""
Evaluate this message for safety and crisis risk.

CONVERSATION HISTORY:
{history}

CURRENT MESSAGE: "{user_text}"

Check for: suicidal ideation, self-harm intent, life-threatening medical emergency.
- If EMERGENCY detected: use the emergency_call tool, then respond with crisis resources.
- If NOT emergency: respond with "SAFE: [one sentence about what the user needs]"
        """,
        expected_output="Either crisis response with resources, OR 'SAFE: [brief summary]'",
        agent=safety_agent,
    )

    doctor_task = Task(
        description=f"""
CRITICAL INSTRUCTION: Do NOT ask the user for their city or location.
Do NOT offer to find nearby doctors or specialists. Just give medical guidance.

Provide medical guidance for this user.

CONVERSATION HISTORY:
{history}

CURRENT MESSAGE: "{user_text}"

Safety check result is in your context. The user is not in crisis.

RESPONSE: 2 paragraphs only.
- Paragraph 1: Medical info — causes, home care, warning signs. Be specific.
- Paragraph 2: One empathetic sentence. One sentence to see a doctor if needed.

FORBIDDEN — do not include any of these in your response:
- "Would you like me to find" 
- "nearby doctors"
- "reply with your city"
- "city name"
- Any offer to search or find professionals
        """,
        expected_output="2 paragraphs: medical info + brief empathy/advice. No location offers whatsoever.",
        agent=doctor_agent,
        context=[safety_task],
    )

    # Check if a location is present in current message or recent history
    # so we can tell the agent explicitly whether to search or not
    _has_location = _detect_location_followup(user_text, session) is not None or                     any(w in user_text.lower() for w in [
                        "nagpur","mumbai","pune","delhi","hyderabad","bangalore",
                        "bengaluru","chennai","kolkata","lucknow","kanpur",
                        "wadi","banjara","koregaon","andheri","bandra",
                    ])

    if _has_location:
        _tool_instruction = f"""
IMPORTANT — TOOL SEARCH REQUIRED:
A location was mentioned. You MUST call find_nearby_therapists NOW with the
location from the current message: "{user_text}"
After calling the tool, your ENTIRE response must be:
- ONE warm sentence (e.g. "Here are some mental health professionals near you:")
- The FULL clinic list exactly as returned by the tool (names, addresses, ratings)
- The helpline numbers at the end
Do NOT give a coping technique. Do NOT ask for city again. ONLY show the clinics."""
    else:
        _tool_instruction = """
No location provided yet. Do NOT call any tool.
Response structure:
- Paragraph 1 (10%): ONE sentence validating their feeling.
- Paragraph 2 (80%): ONE specific coping technique. Be specific.
- Paragraph 3 (10%): "Would you like me to find a therapist near you?
  If yes, reply with your city name." """

    therapist_task = Task(
        description=f"""
Provide mental health support for this user.

CONVERSATION HISTORY:
{history}

CURRENT MESSAGE: "{user_text}"

Safety check result is in your context. The user is not in crisis.

{_tool_instruction}

RULES ALWAYS:
- Maximum 3 paragraphs total
- Do NOT repeat a coping technique already given in conversation history
- Do NOT repeat the user's words back to them
        """,
        expected_output="If location given: ONLY the clinic list from tool. If no location: support + ask for city.",
        agent=therapist_agent,
        context=[safety_task],
    )

    # ── Step 5: Assemble crew ──────────────────────────────────────────────────
    agents, tasks = _build_crew_for_intent(
        intent=intent_result.intent,
        safety_agent=safety_agent,
        doctor_agent=doctor_agent,
        therapist_agent=therapist_agent,
        safety_task=safety_task,
        doctor_task=doctor_task,
        therapist_task=therapist_task,
    )

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )

    # ── Step 6: Run crew with LangSmith tracing ──────────────────────────────
    # CRITICAL: litellm.success_callback must be set INSIDE the executor
    # thread — not in the main async thread. Python's litellm module
    # uses threading.local() for some state, so setting callbacks in the
    # main thread doesn't propagate to the thread where crew.kickoff() runs.
    def _kickoff_with_langsmith_tracing():
        """Run crew.kickoff() with LangSmith callback active in this thread."""
        try:
            import litellm as _litellm
            # Register langsmith callback in THIS thread where LLM calls happen
            if "langsmith" not in (_litellm.success_callback or []):
                _litellm.success_callback = list(_litellm.success_callback or []) + ["langsmith"]
        except Exception:
            pass  # non-fatal
        return crew.kickoff()

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, _kickoff_with_langsmith_tracing
        )
        response_text = str(result.raw) if hasattr(result, 'raw') else str(result)
        response_text = _clean_crew_response(response_text)
        response_text = _trim_response(response_text)
    except Exception as e:
        logger.exception("Crew execution failed for user %s: %s", user_id, e)
        response_text = (
            "I'm sorry, I had trouble processing your message. "
            "Please try again. If this is urgent, call 112."
        )

    escalated = _detect_escalation(tasks)

    return ChatResponse(
        text=response_text,
        intent=intent_result.intent,
        escalated=escalated,
    )


def _build_crew_for_intent(
    intent, safety_agent, doctor_agent, therapist_agent,
    safety_task, doctor_task, therapist_task,
):
    if intent == Intent.MEDICAL:
        return ([safety_agent, doctor_agent], [safety_task, doctor_task])
    elif intent == Intent.THERAPY:
        return ([safety_agent, therapist_agent], [safety_task, therapist_task])
    elif intent == Intent.MIXED:
        therapist_task.context = [safety_task, doctor_task]
        return (
            [safety_agent, doctor_agent, therapist_agent],
            [safety_task, doctor_task, therapist_task],
        )
    else:
        return ([safety_agent, therapist_agent], [safety_task, therapist_task])


def _detect_escalation(tasks) -> bool:
    try:
        safety_output = str(tasks[0].output) if tasks[0].output else ""
        return "EMERGENCY_CALL_PLACED" in safety_output
    except Exception:
        return False