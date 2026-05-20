"""
observability/tracer.py — LangSmith tracing for SafeSpace AI

HOW PARENT-CHILD NESTING WORKS:
=================================
LiteLLM (used by CrewAI for all LLM calls) reads the env var
LANGCHAIN_PARENT_RUN_ID before each API call. If set, it attaches
that LLM run as a child of the specified parent run ID in LangSmith.

Flow:
  1. trace_request() creates a parent run via client.create_run()
     → gets a run_id (UUID)
  2. Sets os.environ["LANGCHAIN_PARENT_RUN_ID"] = run_id
  3. CrewAI runs → LiteLLM makes LLM calls → each call sees
     LANGCHAIN_PARENT_RUN_ID → sends to LangSmith as child run
  4. trace_request() clears the env var + closes parent run

Result in LangSmith:
  safespace_request                     ← parent (our run)
      ├── llama-3.3-70b (safety)        ← child (LiteLLM auto-nested) ✅
      ├── llama-3.3-70b (doctor)        ← child (LiteLLM auto-nested) ✅
      └── llama-3.3-70b (therapist)     ← child (LiteLLM auto-nested) ✅

WHY NOT @traceable:
  @traceable uses contextvars for propagation.
  CrewAI.kickoff() runs in a thread executor with asyncio.run()
  which creates a new event loop → contextvars are NOT inherited
  → @traceable parent context is invisible to LiteLLM callbacks.

  LANGCHAIN_PARENT_RUN_ID is a plain os.environ key → always visible
  across threads and event loops → reliable propagation.
"""

import logging
import os
import time
import uuid
from contextlib import contextmanager
from typing import Optional

from core.config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

# ── Setup ─────────────────────────────────────────────────────────────────────

def _setup_langsmith() -> None:
    if settings.langsmith_configured:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"]    = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"]    = settings.langsmith_project
        os.environ["LANGSMITH_API_KEY"]    = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"]    = settings.langsmith_project
        os.environ["LANGSMITH_TRACING"]    = "true"
        logger.info("LangSmith tracing enabled → project: %s",
                    settings.langsmith_project)
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"]    = "false"
        logger.info("LangSmith tracing disabled (no API key)")


_setup_langsmith()


# ── Main tracing context ───────────────────────────────────────────────────────

@contextmanager
def trace_request(user_id: str, channel: str = "unknown",
                  intent: Optional[str] = None, **metadata):
    """
    Context manager that creates a parent LangSmith run and sets
    LANGCHAIN_PARENT_RUN_ID so CrewAI/LiteLLM LLM calls nest under it.

    Usage in engine.py:
        with trace_request(user_id, channel="whatsapp") as run_id:
            result = await run_crew(...)

    LangSmith dashboard will show:
        safespace_request
            ├── groq/llama (safety check)
            ├── groq/llama (doctor/therapist)
            └── ... all LLM calls nested ✅
    """
    if not settings.langsmith_configured:
        yield None
        return

    start_time = time.time()
    run_id     = str(uuid.uuid4())
    safe_uid   = _anonymise_user_id(user_id)

    try:
        from langsmith import Client
        client = Client(api_key=settings.langsmith_api_key)
    except Exception as e:
        logger.debug("LangSmith client init failed: %s", e)
        yield None
        return

    # ── Create parent run ──────────────────────────────────────────────────
    try:
        client.create_run(
            id=run_id,
            name="safespace_request",
            run_type="chain",
            project_name=settings.langsmith_project,
            inputs={
                "user_id":  safe_uid,
                "channel":  channel,
                **metadata,
            },
            tags=["safespace", "v2", channel],
            start_time=None,
        )
    except Exception as e:
        logger.debug("LangSmith create_run failed (non-fatal): %s", e)
        yield None
        return

    # ── KEY: set parent run ID so LiteLLM nests its calls here ────────────
    prev_parent = os.environ.get("LANGCHAIN_PARENT_RUN_ID")
    os.environ["LANGCHAIN_PARENT_RUN_ID"] = run_id

    try:
        yield run_id  # caller can use this for log_intent()
        
        latency_ms = int((time.time() - start_time) * 1000)
        _close_run(client, run_id, settings.langsmith_project,
                   outputs={
                       "status":     "success",
                       "latency_ms": latency_ms,
                       "intent":     intent or "unknown",
                   })

    except Exception as exc:
        latency_ms = int((time.time() - start_time) * 1000)
        _close_run(client, run_id, settings.langsmith_project,
                   outputs={"status": "error", "latency_ms": latency_ms},
                   error=str(exc))
        raise

    finally:
        # ── Always restore previous parent ID (thread-safety) ─────────────
        if prev_parent is not None:
            os.environ["LANGCHAIN_PARENT_RUN_ID"] = prev_parent
        elif "LANGCHAIN_PARENT_RUN_ID" in os.environ:
            del os.environ["LANGCHAIN_PARENT_RUN_ID"]


def _close_run(client, run_id: str, project: str,
               outputs: dict, error: Optional[str] = None) -> None:
    try:
        client.update_run(
            run_id=run_id,
            outputs=outputs,
            error=error,
            end_time=None,
        )
    except Exception as e:
        logger.debug("LangSmith update_run failed (non-fatal): %s", e)


# ── Intent logging ─────────────────────────────────────────────────────────────

def log_intent(user_id: str, intent: str, confidence: float,
               channel: str = "unknown",
               run_id: Optional[str] = None) -> None:
    """
    Log intent classification to LangSmith dataset.
    Also adds intent as metadata to the active parent run if run_id given.
    """
    if not settings.langsmith_configured:
        return
    try:
        from langsmith import Client
        client = Client(api_key=settings.langsmith_api_key)

        # Attach intent to parent run as output update
        if run_id:
            try:
                client.update_run(
                    run_id=run_id,
                    outputs={
                        "intent":     intent,
                        "confidence": confidence,
                        "channel":    channel,
                    },
                )
            except Exception:
                pass

        # Also log to dataset for analytics
        try:
            client.create_example(
                inputs={
                    "user_id": _anonymise_user_id(user_id),
                    "channel": channel,
                },
                outputs={
                    "intent":     intent,
                    "confidence": confidence,
                },
                dataset_name="safespace_intent_classification",
            )
        except Exception:
            pass  # dataset may not exist yet — non-fatal

    except Exception as e:
        logger.debug("LangSmith log_intent failed (non-fatal): %s", e)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _anonymise_user_id(user_id: str) -> str:
    """Privacy-safe user ID for LangSmith."""
    if "whatsapp:" in user_id:
        number = user_id.replace("whatsapp:", "")
        if len(number) > 6:
            return f"whatsapp:{number[:4]}*******{number[-3:]}"
    if user_id.startswith("web_"):
        return f"{user_id[:8]}****"
    return user_id[:8] + "****"