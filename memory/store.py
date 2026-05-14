"""
memory/store.py

SQLite-backed persistent session memory.
Replaces the in-memory dict from SafeSpace 1.0 —
sessions now survive server restarts.
"""

import json
import logging
import aiosqlite
from core.config import get_settings
from core.schemas import UserSession, SessionMessage

logger = logging.getLogger(__name__)
settings = get_settings()

DB_PATH = settings.sqlite_db_path


async def init_db() -> None:
    """Create tables if they don't exist. Called once at startup."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                user_id     TEXT PRIMARY KEY,
                messages    TEXT NOT NULL DEFAULT '[]',
                updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()
    logger.info("DB initialised: %s", DB_PATH)


async def get_session(user_id: str) -> UserSession:
    """Load a user's session. Returns empty session if first-time user."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT messages FROM sessions WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()

    if not row:
        return UserSession(user_id=user_id)

    messages = [SessionMessage(**m) for m in json.loads(row[0])]
    return UserSession(
        user_id=user_id,
        messages=messages,
        message_count=len(messages),
    )


async def save_message(user_id: str, role: str, content: str) -> None:
    """
    Append a message to the user's session.
    Trims to SESSION_MAX_MESSAGES to avoid unbounded growth.
    """
    session = await get_session(user_id)
    session.messages.append(SessionMessage(role=role, content=content))

    # Keep only the most recent N messages
    max_msgs = settings.session_max_messages
    if len(session.messages) > max_msgs:
        session.messages = session.messages[-max_msgs:]

    messages_json = json.dumps([m.model_dump() for m in session.messages])

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO sessions (user_id, messages, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                messages   = excluded.messages,
                updated_at = CURRENT_TIMESTAMP
        """, (user_id, messages_json))
        await db.commit()


async def clear_session(user_id: str) -> None:
    """Wipe a user's session — useful for testing."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        await db.commit()
    logger.info("Session cleared for user %s", user_id)


def format_history_for_llm(session: UserSession) -> str:
    """
    Formats session history as a readable string for LLM context.
    Passed into agent prompts so agents have conversation memory.
    """
    if not session.messages:
        return "No prior conversation."
    lines = []
    for msg in session.messages[-10:]:  # last 10 messages max in prompt
        prefix = "User" if msg.role == "user" else "SafeSpace"
        lines.append(f"{prefix}: {msg.content}")
    return "\n".join(lines)