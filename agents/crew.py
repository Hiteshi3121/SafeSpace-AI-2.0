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
    Detects when user replies with a city after AI asked
    "Would you like me to find a therapist near you? Reply with your city."

    Bypasses intent classification entirely and directly calls the maps tool.
    Returns formatted results string, or None if not a location follow-up.
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

    # Did the AI ask for a city?
    asked_for_city = any(phrase in last_ai.lower() for phrase in [
        "reply with your city",
        "reply with your city name",
        "tell me your city",
        "what city are you in",
    ])
    if not asked_for_city:
        return None

    # Is user's reply a city (not a new question)?
    user_lower = user_text.lower().strip()
    skip_if_starts = ["what", "how", "why", "tell me", "i want", "i need",
                      "can you", "please", "i feel", "i have", "i am not"]
    if any(user_lower.startswith(p) for p in skip_if_starts):
        return None

    # User said no
    if user_lower in ["no", "nope", "na", "nahi", "not now", "no thanks"]:
        return "No problem! Let me know if you need anything else. 🌿"

    # Extract location — strip "yes," prefix variations
    location = user_lower
    for prefix in ["yes,", "yes ", "sure,", "sure ", "ok,", "ok ", "haan,", "haan "]:
        if location.startswith(prefix):
            location = location[len(prefix):].strip()

    if not location or len(location) < 2:
        return None

    location = location.title()

    # Call therapist finder directly — no LLM needed
    from tools.maps_tool import find_therapists_tool
    logger.info("Location follow-up detected: searching therapists near '%s'", location)
    return find_therapists_tool._run(location=location)



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
Provide medical guidance for this user.

CONVERSATION HISTORY:
{history}

CURRENT MESSAGE: "{user_text}"

Safety check result is in your context. The user is not in crisis.
Follow your guardrails: never diagnose, recommend seeing a doctor for serious concerns.

RESPONSE STRUCTURE (strictly follow this split):
- Paragraph 1 (80%): Medical information — possible causes, practical home care,
  warning signs to watch, when to see a doctor. Be specific, not generic.
- Paragraph 2 (10%): ONE brief empathetic sentence only. Nothing more.
- Paragraph 3 (10%): "Would you like me to find nearby doctors or specialists?
  If yes, reply with your city name." — say this ONLY if relevant.

RULES:
- Maximum 3 paragraphs total
- No repeating the user's symptoms back
- No filler like "I understand how you feel" taking up the main space
        """,
        expected_output="Medical guidance: 80% medical info, 10% empathy, 10% doctor search offer.",
        agent=doctor_agent,
        context=[safety_task],
    )

    therapist_task = Task(
        description=f"""
Provide mental health support for this user.

CONVERSATION HISTORY:
{history}

CURRENT MESSAGE: "{user_text}"

Safety check result is in your context. The user is not in crisis.

RESPONSE STRUCTURE (strictly follow this split):
- Paragraph 1 (10%): ONE sentence validating their feeling. Brief only.
- Paragraph 2 (80%): Practical help — specific coping technique, CBT reframing,
  breathing exercise, or ONE focused actionable step. Be specific, not generic.
- Paragraph 3 (10%): "Would you like me to find a therapist near you?
  If yes, reply with your city name."

RULES:
- Maximum 3 paragraphs total
- Do NOT use more than one sentence for validation
- Do NOT repeat the user's words back to them
- Pick ONE best suggestion, not multiple options
        """,
        expected_output="Support: 10% validation, 80% practical help, 10% therapist search offer.",
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

    # ── Step 6: Run crew ───────────────────────────────────────────────────────
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, crew.kickoff
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