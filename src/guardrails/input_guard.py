"""
Input guardrails for the Agentic RAG v2 pipeline.

Validates and sanitizes user input before it reaches the agent,
acting as a fast, zero-cost Layer 1 defense against Prompt Injection
and DoS attacks via oversized inputs.
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Limits
MAX_INPUT_LENGTH = 2000
MIN_INPUT_LENGTH = 1

# Advanced prompt injection & RAG bypass patterns (Portuguese + English)
INJECTION_PATTERNS = [
    # Classic Jailbreaks
    r"ignore\s+(previous|all|above)\s+(instructions|prompts)",
    r"ignor[ea]\s+(as\s+)?instruções\s+(anteriores|acima)",
    r"you\s+are\s+now\s+a",
    r"você\s+agora\s+é\s+um",
    r"system\s*:\s*",
    r"<\s*system\s*>",
    r"jailbreak",
    r"DAN\s+mode",
    
    # RAG/System Leakage Attempts
    r"(what\s+is|tell\s+me)\s+(your\s+)?(system\s+prompt|initial\s+instructions)",
    r"(qual\s+[ée]|me\s+diga|me\s+mostre)\s+(o\s+seu\s+)?(prompt|instruções\s+iniciais|regras)",
    r"(forget|ignore)\s+(the\s+)?(context|documents|reference)",
    r"(esqueça|ignore)\s+(o\s+)?(contexto|documentos?|referências?)",
]

_compiled_patterns = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


class InputGuardError(Exception):
    """Raised when input fails validation."""

    def __init__(self, message: str, user_message: str):
        super().__init__(message)
        self.user_message = user_message


def _remove_control_characters(text: str) -> str:
    """Removes invisible control characters often used to bypass regex filters."""
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")


def validate_input(text: str) -> str:
    """
    Validate and sanitize user input.

    Args:
        text: Raw user input.

    Returns:
        Cleaned and sanitized text if valid.

    Raises:
        InputGuardError: If input fails validation.
    """
    if not text:
        raise InputGuardError(
            "Empty input",
            "Por favor, envie uma mensagem com sua pergunta. 😊"
        )

    # 1. Sanitize: Remove invisible control characters and strip whitespace
    cleaned = _remove_control_characters(text).strip()

    # 2. Length check: Minimum
    if len(cleaned) < MIN_INPUT_LENGTH:
        logger.warning("🛡️ Input guard: empty message rejected")
        raise InputGuardError(
            "Empty input",
            "Por favor, envie uma mensagem com sua pergunta. 😊",
        )

    # 3. Length check: Maximum (DoS prevention)
    if len(cleaned) > MAX_INPUT_LENGTH:
        logger.warning(f"🛡️ Input guard: message too long ({len(cleaned)} chars)")
        raise InputGuardError(
            f"Input too long: {len(cleaned)} chars",
            f"Sua mensagem está muito longa ({len(cleaned)} caracteres). "
            f"O limite é de {MAX_INPUT_LENGTH} caracteres. "
            f"Por favor, tente reformular de forma mais concisa. ✂️",
        )

    # 4. Pattern Matching: Prompt injection check
    for pattern in _compiled_patterns:
        if pattern.search(cleaned):
            logger.warning(f"🛡️ Input guard: potential injection detected: '{cleaned[:50]}...'")
            raise InputGuardError(
                "Potential prompt injection",
                "Desculpe, não consigo processar essa solicitação. "
                "Meu objetivo é apenas tirar dúvidas técnicas sobre os equipamentos da Visuri. 🔒",
            )

    return cleaned
