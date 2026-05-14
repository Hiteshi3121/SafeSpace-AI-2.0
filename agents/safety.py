"""
agents/safety.py

SafetyAgent — the first agent that evaluates EVERY message.

ARCHITECTURE ROLE:
==================
In a hierarchical CrewAI crew, the SafetyAgent acts like a triage nurse.
Before Doctor or Therapist respond, SafetyAgent checks:
  1. Is this a genuine emergency? (suicidal ideation, self-harm, medical emergency)
  2. Does the user need immediate escalation?

If yes → it uses the EmergencyCallTool and the crew stops routing to other agents.
If no  → it signals the crew to continue with Doctor/Therapist.

WHY HIERARCHICAL PROCESS:
==========================
With Process.hierarchical, CrewAI creates a "manager" LLM that coordinates
which agent gets which task. This means:
  - SafetyAgent can short-circuit the flow if emergency is detected
  - Doctor and Therapist only run when Safety clears the message
  - We don't need to manually write if/else routing logic

CREWAI AGENT ANATOMY:
======================
  role       → The agent's job title (used in crew coordination)
  goal       → What the agent is trying to achieve
  backstory  → Persona/context that shapes how the LLM responds
  tools      → List of Tool instances the agent can use
  llm        → Which model to use
  verbose    → Whether to log the agent's internal reasoning
"""

import logging
from crewai import Agent
from core.config import get_settings
from tools.emergency_tool import emergency_call_tool

logger = logging.getLogger(__name__)
settings = get_settings()

SAFETY_AGENT_ROLE = "Crisis Detection & Safety Specialist"

SAFETY_AGENT_GOAL = """
Evaluate every user message for safety concerns before any other agent responds.
Detect genuine emergencies (suicidal ideation, self-harm intent, life-threatening medical emergencies).
Trigger the emergency call tool ONLY for verified emergencies.
For non-emergencies, provide a brief safety clearance and let the appropriate specialist respond.
"""

SAFETY_AGENT_BACKSTORY = """
You are SafeSpace's crisis detection specialist with training in psychological first aid
and medical triage. You have reviewed thousands of mental health crisis cases.

You are CALM, NON-ALARMIST, and ACCURATE. You do not over-escalate.
You distinguish between:
  - Someone venting frustration ("I want to kill my sister") → NOT an emergency
  - Someone expressing genuine suicidal ideation ("I've been thinking about ending it") → EMERGENCY
  - General anxiety or depression → NOT an emergency (route to TherapistAgent)
  - Chest pain + breathlessness in an elderly user → possible EMERGENCY

When you detect a genuine emergency:
  1. Use the emergency_call tool immediately
  2. Respond with crisis resources (iCall: 9152987821, Vandrevala Foundation: 1860-2662-345)
  3. Stay with the user — keep them talking

When it is NOT an emergency:
  Simply respond: "SAFE: [one sentence summary of what the user needs]"
  This tells the crew manager to route to the appropriate specialist.
"""


def create_safety_agent() -> Agent:
    """
    Factory function that creates and returns the SafetyAgent.
    Using a factory instead of a module-level singleton because
    CrewAI agents hold state during a crew run — we want a fresh
    instance per request in production.
    """
    return Agent(
        role=SAFETY_AGENT_ROLE,
        goal=SAFETY_AGENT_GOAL,
        backstory=SAFETY_AGENT_BACKSTORY,
        tools=[emergency_call_tool],
        llm=f"groq/llama-3.3-70b-versatile",
        verbose=settings.is_development,   # Only log internals in dev
        allow_delegation=False,             # Safety agent never delegates
        max_iter=3,                         # Limit reasoning loops
        memory=False,                       # Memory handled by our SQLite layer
    )