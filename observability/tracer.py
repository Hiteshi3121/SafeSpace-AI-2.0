"""
observability/tracer.py

LangSmith tracing for SafeSpace AI.

WHAT GETS TRACED:
==================
Every request through handle_request() is wrapped in a LangSmith run.
Each run captures:
  - user_id (anonymised channel + number)
  - intent (MEDICAL / THERAPY / MIXED)
  - which agents ran
  - total latency
  - escalation status
  - any errors

WHY LANGSMITH OVER CREWAI'S BUILT-IN TRACING:
===============================================
CrewAI's tracing (app.crewai.com) is great for debugging agent internals
but it's ephemeral (24hr links) and not production-grade.

LangSmith gives us:
  - Persistent trace history
  - Custom metadata filtering ("show me all MEDICAL requests today")
  - Custom evaluators (safety score, empathy score)
  - Latency tracking per agent
  - Error rate monitoring
  - Dataset creation from real traces → use for future fine-tuning

HOW LANGCHAIN_TRACING_V2 WORKS WITH CREWAI:
=============================================
When LANGCHAIN_TRACING_V2=true is set in env, LiteLLM (which CrewAI uses
for all LLM calls) automatically sends every LLM call to LangSmith.
We don't need to instrument individual agent calls — the env var does it.

Our trace_request() context manager adds a parent span that groups all
LLM calls from a single user request under one trace in the dashboard.
"""

import logging
import os
import time
from contextlib import contextmanager
from typing import Any

from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _setup_langsmith() -> None:
    """
    Set LangSmith env vars at module load time.
    Must happen before any LiteLLM/CrewAI imports so they pick it up.
    """
    if settings.langsmith_configured:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        # Also set LANGSMITH_ prefixed vars (newer SDK versions)
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_TRACING"] = "true"
        logger.info("LangSmith tracing enabled → project: %s", settings.langsmith_project)
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
        logger.info("LangSmith tracing disabled (no API key set)")


# Run once at import time — before any LLM calls happen
_setup_langsmith()


@contextmanager
def trace_request(user_id: str, **metadata):
    """
    Context manager that creates a parent LangSmith run for one user request.

    All LLM calls made inside this context (by CrewAI agents via LiteLLM)
    are automatically nested under this parent run in the dashboard.

    Usage:
        with trace_request(user_id="whatsapp:+91...", channel="whatsapp"):
            result = await run_crew(...)

    In LangSmith dashboard you'll see:
        safespace_request
            ├── intent_classification  (direct Groq call)
            ├── safety_agent_task      (CrewAI → LiteLLM → Groq)
            └── doctor_agent_task      (CrewAI → LiteLLM → Groq)
    """
    if not settings.langsmith_configured:
        yield
        return

    start_time = time.time()

    try:
        from langsmith import traceable
        # Anonymise user_id for privacy — keep channel type, hash the number
        safe_user_id = _anonymise_user_id(user_id)

        # We use the low-level Client to create a run with full metadata
        from langsmith import Client
        client = Client()

        run_id = None
        try:
            import uuid
            run_id = str(uuid.uuid4())
            client.create_run(
                id=run_id,
                name="safespace_request",
                run_type="chain",
                project_name=settings.langsmith_project,
                inputs={"user_id": safe_user_id, **metadata},
                tags=["safespace", "v2"],
            )
        except Exception as e:
            logger.debug("LangSmith run creation failed (non-fatal): %s", e)
            yield
            return

        try:
            yield
            latency_ms = int((time.time() - start_time) * 1000)
            # Update run with outputs on success
            try:
                client.update_run(
                    run_id=run_id,
                    outputs={"latency_ms": latency_ms, "status": "success"},
                    end_time=None,
                )
            except Exception:
                pass

        except Exception as exc:
            latency_ms = int((time.time() - start_time) * 1000)
            try:
                client.update_run(
                    run_id=run_id,
                    error=str(exc),
                    outputs={"latency_ms": latency_ms, "status": "error"},
                )
            except Exception:
                pass
            raise

    except ImportError:
        logger.warning("langsmith package not installed — tracing skipped")
        yield
    except Exception as e:
        logger.warning("LangSmith trace setup failed (non-fatal): %s", e)
        yield


def log_intent(user_id: str, intent: str, confidence: float, channel: str = "unknown") -> None:
    """
    Log intent classification result as a LangSmith feedback event.
    This feeds into our custom evaluators in the dashboard.

    Call this after classify_intent() returns — gives us intent distribution
    analytics over time (what % of users ask medical vs therapy questions).
    """
    if not settings.langsmith_configured:
        return
    try:
        from langsmith import Client
        client = Client()
        # Log as a dataset example for future evaluation
        client.create_example(
            inputs={"user_id": _anonymise_user_id(user_id), "channel": channel},
            outputs={"intent": intent, "confidence": confidence},
            dataset_name="safespace_intent_classification",
        )
    except Exception as e:
        logger.debug("LangSmith intent log failed (non-fatal): %s", e)


def _anonymise_user_id(user_id: str) -> str:
    """
    Privacy-safe user ID for LangSmith.
    'whatsapp:+919876543210' → 'whatsapp:+91*******210'
    'web_042d7192'           → 'web_042d****'
    """
    if "whatsapp:" in user_id:
        number = user_id.replace("whatsapp:", "")
        if len(number) > 6:
            return f"whatsapp:{number[:4]}*******{number[-3:]}"
    if user_id.startswith("web_"):
        return f"{user_id[:8]}****"
    return user_id[:8] + "****"