"""
Output guardrails for the Agentic RAG v2 pipeline.

Validates and sanitizes agent output before sending to the user.
"""

import logging

logger = logging.getLogger(__name__)

# Telegram message limit
TELEGRAM_MAX_LENGTH = 4096

FALLBACK_EMPTY_RESPONSE = (
    "Desculpe, ocorreu um erro ao gerar a resposta. "
    "Por favor, tente novamente. 🔄"
)


def sanitize_output(text: str) -> str:
    """
    Sanitize and validate agent output before sending to the user.

    - Handles empty/None responses
    - Truncates responses exceeding Telegram's 4096 character limit

    Args:
        text: Raw agent output.

    Returns:
        Sanitized output string.
    """
    # Handle None or empty
    if not text or not text.strip():
        logger.warning("🛡️ Output guard: empty response, using fallback")
        return FALLBACK_EMPTY_RESPONSE

    cleaned = text.strip()

    # Truncate if too long for Telegram
    if len(cleaned) > TELEGRAM_MAX_LENGTH:
        logger.warning(
            f"🛡️ Output guard: truncating response "
            f"({len(cleaned)} → {TELEGRAM_MAX_LENGTH} chars)"
        )
        # Cut at last complete sentence before limit
        truncated = cleaned[: TELEGRAM_MAX_LENGTH - 50]
        last_period = truncated.rfind(".")
        if last_period > TELEGRAM_MAX_LENGTH // 2:
            truncated = truncated[: last_period + 1]

        truncated += "\n\n_(Resposta truncada por ser muito longa)_"
        return truncated

    return cleaned
