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
from datetime import datetime, timezone

from google.cloud import firestore

from src.config import GCP_PROJECT_ID, FIRESTORE_COLLECTION

logger = logging.getLogger(__name__)

# Module-level singleton using AsyncClient to prevent Event Loop blocking
_db: firestore.AsyncClient | None = None


def get_firestore_client() -> firestore.AsyncClient:
    """
    Get or initialize the asynchronous Firestore client (singleton).

    Uses Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS).
    """
    global _db
    if _db is None:
        if not GCP_PROJECT_ID:
            raise ValueError(
                "GCP_PROJECT_ID not set. Cannot connect to Firestore."
            )
        _db = firestore.AsyncClient(project=GCP_PROJECT_ID)
        logger.info(f"🔥 Connected to Async Firestore (project: {GCP_PROJECT_ID})")
    return _db


async def save_message(chat_id: str, role: str, content: str) -> None:
    """
    Asynchronously save a message to the conversation history.

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
        
        # Non-blocking database write
        await doc_ref.set({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc),
        })
        logger.debug(f"💾 Saved {role} message for chat {chat_id}")
    except Exception as e:
        logger.error(f"❌ Failed to save message to Firestore: {e}")


async def get_history(chat_id: str, limit: int = 5) -> list[dict]:
    """
    Asynchronously retrieve recent conversation history for a chat.

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
        
        messages = []
        # Non-blocking database read
        async for doc in messages_ref.stream():
            doc_data = doc.to_dict()
            messages.append({
                "role": doc_data["role"], 
                "content": doc_data["content"]
            })
            
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
    Assistant responses are truncated to 150 chars to avoid prompt dilution.
    """
    if not messages:
        return ""

    lines = []
    for msg in messages:
        role_label = "Usuário" if msg["role"] == "user" else "Assistente"
        content = msg["content"]
        
        # Truncate long assistant responses to keep prompt lean
        if msg["role"] == "assistant" and len(content) > 150:
            content = content[:147] + "..."
            
        lines.append(f"{role_label}: {content}")

    return "\n".join(lines)
