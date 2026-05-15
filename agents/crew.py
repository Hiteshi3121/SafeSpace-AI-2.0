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
    Detects if the user is responding with a location after the AI asked
    "Would you like me to find nearby doctors/therapists? Reply with your city."
    
    Returns the maps search result string if it's a location follow-up,
    or None if it's a regular message.
    """
    if not session.messages or len(session.messages) < 2:
        return None

    # Check if the last AI message asked for a location
    last_ai_messages = [
        m.content for m in reversed(session.messages)
        if m.role == "assistant"
    ]
    if not last_ai_messages:
        return None

    last_ai_msg = last_ai_messages[0].lower()
    asked_for_location = any(phrase in last_ai_msg for phrase in [
        "reply with your city",
        "reply with your city name",
        "tell me your city",
        "which city are you in",
        "what city are you in",
    ])

    if not asked_for_location:
        return None

    # Check if user's reply looks like a location (not a question or long sentence)
    user_lower = user_text.lower().strip()

    # Skip if it looks like a full question/sentence, not a city reply
    skip_phrases = ["what", "how", "why", "tell me", "i want", "i need",
                    "can you", "please", "find me", "search"]
    if any(user_lower.startswith(p) for p in skip_phrases):
        return None

    # User said "no" → don't search
    if user_lower in ["no", "nope", "nahi", "na", "not now", "no thanks"]:
        return "No problem! Feel free to ask if you need anything else."

    # Extract location from user reply
    # Handle: "yes, nagpur", "yes nagpur", "nagpur", "moosapet hyderabad", "yes moosapet in hyderabad"
    location = user_lower
    for strip_word in ["yes,", "yes ", "sure,", "sure ", "ok,", "ok ", "haan,", "haan "]:
        if location.startswith(strip_word):
            location = location[len(strip_word):].strip()

    if not location or len(location) < 2:
        return None

    location = location.title()  # "nagpur" → "Nagpur"

    # Determine what to search for based on what AI asked
    from tools.maps_tool import find_therapists_tool, find_doctors_tool
    settings_inst = get_settings()
    api_key = getattr(settings_inst, "google_maps_api_key", "")

    if not api_key:
        return (
            f"I'd like to find professionals near {location}, but Google Maps "
            "isn't configured. Please try Practo.com or 1mg.com directly."
        )

    if any(word in last_ai_msg for word in ["therapist", "mental health", "psychiatrist", "counselor"]):
        logger.info("Location follow-up: therapist search for '%s'", location)
        return find_therapists_tool._run(location=location)
    else:
        # Doctor/specialist search — detect specialty from conversation history
        specialty = _infer_specialty_from_history(session)
        logger.info("Location follow-up: doctor search for '%s' specialty='%s'", location, specialty)
        return find_doctors_tool._run(location=location, specialty=specialty)


def _infer_specialty_from_history(session) -> str:
    """
    Uses the LLM to read the conversation history and decide which
    medical specialist is most appropriate for the user's symptoms.
    
    Much better than keyword mapping because:
    - Understands context and combinations of symptoms
    - Handles unusual/rare symptoms the keyword map wouldn't catch
    - Reasons about severity (e.g. "severe chest pain" → cardiologist not GP)
    - Works in any language the user writes in
    - No maintenance needed when new specialties are added
    """
    from groq import Groq
    from core.config import get_settings

    settings = get_settings()

    # Build the conversation summary for the LLM
    recent_messages = session.messages[-8:]  # last 8 messages
    if not recent_messages:
        return "general physician doctor clinic"

    conversation = "\n".join(
        f"{'User' if m.role == 'user' else 'SafeSpace'}: {m.content}"
        for m in recent_messages
    )

    try:
        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical triage assistant. "
                        "Given a conversation, identify the most appropriate medical specialist "
                        "to search for on Google Maps. "
                        "Reply with ONLY a short search query like 'cardiologist' or "
                        "'dermatologist skin specialist' or 'general physician'. "
                        "No explanation. No punctuation. Just the specialty search term."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Based on this conversation, what type of doctor/specialist "
                        f"should I search for on Google Maps?\n\n{conversation}"
                    )
                }
            ],
            max_tokens=20,   # We only need a short specialty name
            temperature=0.1, # Low temperature = consistent, focused answers
        )
        specialty = response.choices[0].message.content.strip().lower()
        logger.info("LLM inferred specialty: '%s'", specialty)
        return specialty if specialty else "general physician doctor clinic"

    except Exception as e:
        logger.warning("LLM specialty inference failed: %s — using default", e)
        return "general physician doctor clinic"


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
    # ── Location follow-up detection ──────────────────────────────────────────
    # Check if previous AI message asked for a city and user is now replying
    # with a location. If so, skip normal intent classification and directly
    # call the maps tool with the provided location.
    location_response = _detect_location_followup(user_text, session)
    if location_response:
        return ChatResponse(
            text=location_response,
            intent=Intent.MEDICAL,
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

RESPONSE STRUCTURE (strictly follow this):
- Paragraph 1 (80% of response): Medical information — possible causes, what it means,
  practical home care steps, warning signs to watch for, when to see a doctor.
  Be specific and useful. This is the MAIN part.
- Paragraph 2 (10% of response): One brief empathetic sentence only. No more.
- Paragraph 3 (10% of response): Ask if they want nearby doctors/specialists.
  Say: "Would you like me to find nearby doctors or specialists for this?
  If yes, reply with your city name."

RULES:
- Maximum 3 paragraphs total
- No repeating the user's symptoms
- No filler phrases like "I understand" or "That must be difficult" taking up space
- The medical content must be specific to their symptoms, not generic
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

RESPONSE STRUCTURE (strictly follow this):
- Paragraph 1 (10% of response): One sentence validating their feeling. Keep it brief.
- Paragraph 2 (80% of response): Practical help — specific coping technique, CBT reframing,
  breathing exercise, actionable step, or ONE focused follow-up question.
  This is the MAIN part. Be specific and useful, not generic.
- Paragraph 3 (10% of response): Offer therapist search.
  Say: "Would you like me to find a therapist near you? If yes, reply with your city."

RULES:
- Maximum 3 paragraphs total
- Do NOT use more than one sentence for validation — warmth should be in the advice, not preamble
- Do NOT repeat the user's words back to them
- Do NOT offer multiple suggestions — pick ONE best one
        """,
        expected_output="Support response: 10% validation, 80% practical help, 10% therapist search offer.",
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