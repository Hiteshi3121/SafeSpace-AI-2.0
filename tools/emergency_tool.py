"""
tools/emergency_tool.py

CrewAI Tool that triggers an emergency phone call via Twilio.

WHY A TOOL (NOT JUST A FUNCTION)?
===================================
In CrewAI, a Tool is something an Agent can CHOOSE to use during its task.
The SafetyAgent doesn't always call this — it uses it only when it detects
a genuine crisis (suicidal ideation, self-harm, medical emergency).

By making it a Tool, we get:
  1. The agent decides WHEN to use it (not hardcoded logic)
  2. LangSmith traces every tool invocation automatically
  3. Easy to mock in tests without touching Twilio

HOW CREWAI TOOLS WORK:
========================
A Tool is a class with:
  - name: str           → what the agent calls it in its reasoning
  - description: str    → what the agent reads to decide if it should use it
  - _run(args): str     → the actual function that executes

The LLM reads 'description' and decides "should I use this tool right now?"
So the description must be clear and action-oriented.
"""

import logging
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from core.config import get_settings


def _get_traceable():
    """Lazy import so tool works even if langsmith not installed."""
    try:
        from langsmith import traceable
        return traceable
    except ImportError:
        return lambda *a, **k: (lambda f: f)

logger = logging.getLogger(__name__)
settings = get_settings()


class EmergencyCallInput(BaseModel):
    """Input schema for the emergency call tool."""
    reason: str = Field(..., description="Brief reason for the emergency call")


class EmergencyCallTool(BaseTool):
    """
    Triggers an emergency phone call to the user's emergency contact.
    Use this ONLY when the user expresses suicidal ideation, self-harm intent,
    or a life-threatening medical emergency.
    """
    name: str = "emergency_call"
    description: str = (
        "Triggers an emergency phone call to the user's emergency contact. "
        "Use ONLY for genuine emergencies: suicidal ideation, self-harm intent, "
        "or life-threatening medical situations. Do NOT use for general distress."
    )
    args_schema: type[BaseModel] = EmergencyCallInput

    def _run(self, reason: str) -> str:
        """Execute the emergency call via Twilio — traced in LangSmith."""
        traceable = _get_traceable()

        @traceable(
            name="emergency_call",
            run_type="tool",
            metadata={"reason": reason, "tool": "twilio_voice"},
        )
        def _traced_run():
            return self._run_impl(reason)

        try:
            return _traced_run()
        except Exception:
            return self._run_impl(reason)

    def _run_impl(self, reason: str) -> str:
        """Actual emergency call logic."""
        if not settings.twilio_configured:
            logger.warning("Emergency call requested but Twilio not configured")
            return "EMERGENCY_NOTED: Twilio not configured — log this incident manually."

        if not settings.emergency_contact:
            logger.warning("Emergency call requested but EMERGENCY_CONTACT not set")
            return "EMERGENCY_NOTED: No emergency contact configured."

        try:
            from twilio.rest import Client
            client = Client(settings.twilio_account_sid, settings.twilio_auth_token)

            call = client.calls.create(
                to=settings.emergency_contact,
                from_=settings.twilio_from_number,
                twiml=f"""
                    <Response>
                        <Say voice="alice">
                            This is an automated alert from SafeSpace AI.
                            A user may need immediate help. Reason: {reason}.
                            Please check on them immediately.
                        </Say>
                    </Response>
                """,
            )

            logger.info("Emergency call placed. SID: %s | Reason: %s", call.sid, reason)
            return f"EMERGENCY_CALL_PLACED: SID={call.sid}"

        except Exception as e:
            logger.exception("Emergency call failed: %s", e)
            return f"EMERGENCY_CALL_FAILED: {str(e)}"


# Singleton instance — imported directly by SafetyAgent
emergency_call_tool = EmergencyCallTool()