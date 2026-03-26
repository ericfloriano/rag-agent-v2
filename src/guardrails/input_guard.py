"""
Input guardrails for the Agentic RAG v2 pipeline.

Validates and sanitizes user input before it reaches the agent.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Limits
MAX_INPUT_LENGTH = 2000
MIN_INPUT_LENGTH = 1

# Basic prompt injection patterns (Portuguese + English)
INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|above)\s+(instructions|prompts)",
    r"ignor[ea]\s+(as\s+)?instruções\s+(anteriores|acima)",
    r"you\s+are\s+now\s+a",
    r"você\s+agora\s+é\s+um",
    r"system\s*:\s*",
    r"<\s*system\s*>",
    r"jailbreak",
    r"DAN\s+mode",
]

_compiled_patterns = [
    re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS
]


class InputGuardError(Exception):
    """Raised when input fails validation."""

    def __init__(self, message: str, user_message: str):
        super().__init__(message)
        self.user_message = user_message


def validate_input(text: str) -> str:
    """
    Validate and sanitize user input.

    Args:
        text: Raw user input.

    Returns:
        Cleaned text if valid.

    Raises:
        InputGuardError: If input fails validation.
    """
    # Empty check
    cleaned = text.strip()
    if len(cleaned) < MIN_INPUT_LENGTH:
        logger.warning("🛡️ Input guard: empty message rejected")
        raise InputGuardError(
            "Empty input",
            "Por favor, envie uma mensagem com sua pergunta. 😊",
        )

    # Length check
    if len(cleaned) > MAX_INPUT_LENGTH:
        logger.warning(
            f"🛡️ Input guard: message too long ({len(cleaned)} chars)"
        )
        raise InputGuardError(
            f"Input too long: {len(cleaned)} chars",
            f"Sua mensagem está muito longa ({len(cleaned)} caracteres). "
            f"O limite é de {MAX_INPUT_LENGTH} caracteres. "
            f"Por favor, tente reformular de forma mais concisa. ✂️",
        )

    # Prompt injection check
    for pattern in _compiled_patterns:
        if pattern.search(cleaned):
            logger.warning(
                f"🛡️ Input guard: potential injection detected: "
                f"'{cleaned[:50]}...'"
            )
            raise InputGuardError(
                "Potential prompt injection",
                "Desculpe, não consigo processar essa mensagem. "
                "Por favor, faça uma pergunta sobre os equipamentos "
                "ReCARE ou RePAD. 🔒",
            )

    return cleaned
