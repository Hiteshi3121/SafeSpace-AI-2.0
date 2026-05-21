"""
observability/tracer.py — LangSmith tracing for SafeSpace AI

WHAT GETS TRACED (per run):
============================
Input fields:
  user_id      — anonymised (whatsapp:+91*******590 / web_c686****)
  channel      — "whatsapp" or "web"
  message_type — "text" / "image" / "audio"

Output fields (updated after crew runs):
  channel      — "whatsapp" or "web"
  intent       — MEDICAL / THERAPY / MIXED / UNKNOWN
  confidence   — 0.0 to 1.0
  text         — the full response text sent to the user
  escalated    — true/false (was emergency tool called)

LLM Calls tab:
  CrewAI 1.14.4 + LiteLLM sends LLM runs to LangSmith automatically
  via LANGCHAIN_TRACING_V2=true. They appear as separate top-level
  runs (not nested under safespace_request) due to a version
  compatibility limitation between CrewAI 1.14.4 and LangSmith's
  run tree context propagation. The Monitoring dashboard therefore
  shows them in the Runs tab but not the LLM Calls monitoring tab.
  This is expected behaviour for this version combination.
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


@contextmanager
def trace_request(user_id: str, channel: str = "unknown",
                  message_type: str = "text", **metadata):
    """
    Context manager that creates a parent LangSmith run.

    Input logged:
      user_id, channel, message_type

    Output logged (after crew runs via update_trace):
      channel, intent, confidence, text, escalated

    Usage:
        with trace_request(user_id, channel="web",
                           message_type="text") as run_ctx:
            result = await run_crew(...)
            update_trace(run_ctx, result, channel)
    """
    if not settings.langsmith_configured:
        yield {"run_id": None}
        return

    run_id     = str(uuid.uuid4())
    start_time = time.time()
    safe_uid   = _anonymise_user_id(user_id)

    try:
        from langsmith import Client
        client = Client(api_key=settings.langsmith_api_key)
    except Exception as e:
        logger.debug("LangSmith client init failed: %s", e)
        yield {"run_id": None}
        return

    # ── Create parent run — Input fields ──────────────────────────────────
    try:
        client.create_run(
            id=run_id,
            name="safespace_request",
            run_type="chain",
            project_name=settings.langsmith_project,
            inputs={
                "user_id":      safe_uid,
                "channel":      channel,
                "message_type": message_type,
            },
            tags=["safespace", "v2", channel],
        )
    except Exception as e:
        logger.debug("LangSmith create_run failed (non-fatal): %s", e)
        yield {"run_id": None}
        return

    # Set parent run ID so LiteLLM can attach as children (best-effort)
    prev = os.environ.get("LANGCHAIN_PARENT_RUN_ID")
    os.environ["LANGCHAIN_PARENT_RUN_ID"] = run_id

    try:
        yield {"run_id": run_id, "client": client, "start": start_time}

    finally:
        # Restore previous parent
        if prev is not None:
            os.environ["LANGCHAIN_PARENT_RUN_ID"] = prev
        elif "LANGCHAIN_PARENT_RUN_ID" in os.environ:
            del os.environ["LANGCHAIN_PARENT_RUN_ID"]


def update_trace(run_ctx: dict, result, channel: str = "unknown") -> None:
    """
    Called after run_crew() completes to update the parent run
    with Output fields: channel, intent, confidence, text, escalated.
    """
    if not run_ctx or not run_ctx.get("run_id"):
        return
    try:
        latency_ms = int((time.time() - run_ctx["start"]) * 1000)
        run_ctx["client"].update_run(
            run_id=run_ctx["run_id"],
            outputs={
                "intent":     result.intent.value if result.intent else "unknown",
                "channel":    channel,
                "confidence": 1.0,
                "text":       result.text or "",
                "escalated":  result.escalated or False,
                "latency_ms": latency_ms,
            },
        )
    except Exception as e:
        logger.debug("LangSmith update_run failed (non-fatal): %s", e)


def close_trace_error(run_ctx: dict, error: str) -> None:
    """Called when run_crew() raises an exception."""
    if not run_ctx or not run_ctx.get("run_id"):
        return
    try:
        run_ctx["client"].update_run(
            run_id=run_ctx["run_id"],
            error=error,
            outputs={"status": "error"},
        )
    except Exception:
        pass


def log_intent(user_id: str, intent: str, confidence: float,
               channel: str = "unknown",
               run_id: Optional[str] = None) -> None:
    """
    Log intent to the safespace_intent_classification dataset.
    Also updates the active run's output with intent + confidence
    immediately after classify_intent() (before crew runs).
    """
    if not settings.langsmith_configured:
        return
    try:
        from langsmith import Client
        client = Client(api_key=settings.langsmith_api_key)

        # Update parent run output with intent immediately
        if run_id:
            try:
                client.update_run(
                    run_id=run_id,
                    outputs={
                        "channel":    channel,
                        "intent":     intent,
                        "confidence": confidence,
                    },
                )
            except Exception:
                pass

        # Also log to dataset
        try:
            client.create_example(
                inputs={
                    "user_id": _anonymise_user_id(user_id),
                    "channel": channel,
                },
                outputs={"intent": intent, "confidence": confidence},
                dataset_name="safespace_intent_classification",
            )
        except Exception:
            pass

    except Exception as e:
        logger.debug("LangSmith log_intent failed (non-fatal): %s", e)


def _anonymise_user_id(user_id: str) -> str:
    """Privacy-safe user ID: whatsapp:+919*******590, web_c686****"""
    if "whatsapp:" in user_id:
        number = user_id.replace("whatsapp:", "")
        if len(number) > 6:
            return f"whatsapp:{number[:4]}*******{number[-3:]}"
    if user_id.startswith("web_"):
        return f"{user_id[:8]}****"
    return user_id[:8] + "****"