"""
Firestore-based persistent memory for chat sessions.

Stores conversation history per chat_id (Telegram chat ID) in Google Cloud Firestore.
Enables the agent to remember previous interactions across sessions.

Structure:
    chat_sessions/{chat_id}/messages/{auto_id}
        ├── role: "user" | "assistant"
        ├── content: message text
        └── timestamp: server timestamp
"""

import logging
from datetime import datetime

from google.cloud import firestore

from src.config import GCP_PROJECT_ID, FIRESTORE_COLLECTION

logger = logging.getLogger(__name__)

# Module-level singleton
_db: firestore.Client | None = None


def get_firestore_client() -> firestore.Client:
    """
    Get or initialize the Firestore client (singleton).

    Uses Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS).
    """
    global _db
    if _db is None:
        if not GCP_PROJECT_ID:
            raise ValueError(
                "GCP_PROJECT_ID not set. Cannot connect to Firestore."
            )
        _db = firestore.Client(project=GCP_PROJECT_ID)
        logger.info(f"🔥 Connected to Firestore (project: {GCP_PROJECT_ID})")
    return _db


def save_message(chat_id: str, role: str, content: str) -> None:
    """
    Save a message to the conversation history.

    Args:
        chat_id: Telegram chat ID (used as document ID).
        role: "user" or "assistant".
        content: Message text.
    """
    try:
        db = get_firestore_client()
        doc_ref = (
            db.collection(FIRESTORE_COLLECTION)
            .document(str(chat_id))
            .collection("messages")
            .document()
        )
        doc_ref.set({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow(),
        })
        logger.debug(f"💾 Saved {role} message for chat {chat_id}")
    except Exception as e:
        logger.error(f"❌ Failed to save message to Firestore: {e}")


def get_history(chat_id: str, limit: int = 5) -> list[dict]:
    """
    Retrieve recent conversation history for a chat.

    Args:
        chat_id: Telegram chat ID.
        limit: Maximum number of messages to retrieve (default: 5).

    Returns:
        List of dicts with 'role' and 'content' keys, ordered oldest to newest.
    """
    try:
        db = get_firestore_client()
        messages_ref = (
            db.collection(FIRESTORE_COLLECTION)
            .document(str(chat_id))
            .collection("messages")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        docs = messages_ref.stream()
        messages = [
            {"role": doc.to_dict()["role"], "content": doc.to_dict()["content"]}
            for doc in docs
        ]
        # Reverse to get chronological order (oldest first)
        messages.reverse()
        logger.debug(
            f"📜 Retrieved {len(messages)} messages for chat {chat_id}"
        )
        return messages
    except Exception as e:
        logger.error(f"❌ Failed to get history from Firestore: {e}")
        return []


def format_history(messages: list[dict]) -> str:
    """
    Format conversation history for injection into the LLM prompt.

    Args:
        messages: List of message dicts with 'role' and 'content'.

    Returns:
        Formatted string with the conversation history.
    """
    if not messages:
        return ""

    lines = []
    for msg in messages:
        role_label = "Usuário" if msg["role"] == "user" else "Assistente"
        lines.append(f"{role_label}: {msg['content']}")

    return "\n".join(lines)
