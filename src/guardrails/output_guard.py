"""
Output guardrails for the Agentic RAG v2 pipeline.

Validates, sanitizes, and formats agent output before sending to the user.
Prevents platform crashes (e.g., Telegram Markdown errors) and limits size.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Telegram message limits
TELEGRAM_MAX_LENGTH = 4096

FALLBACK_EMPTY_RESPONSE = (
    "Desculpe, ocorreu um erro ao gerar a resposta. "
    "Por favor, tente novamente. 🔄"
)


def _balance_markdown(text: str) -> str:
    """
    Best-effort function to balance unclosed Markdown tags (*, _, `) 
    that might have been left open due to string truncation.
    """
    # Count occurrences of markdown markers
    asterisk_count = text.count("*")
    underscore_count = re.sub(r'[^_]', '', text).count('_') # Avoid counting underscores in URLs
    backtick_count = text.count("`")
    codeblock_count = text.count("```")
    
    # Close code blocks first
    if codeblock_count % 2 != 0:
        text += "\n```"
    # Close inline code
    elif backtick_count % 2 != 0:
        text += "`"
        
    # Close bold/italic (Basic heuristic: if odd, add one to close the last opened)
    if asterisk_count % 2 != 0:
        text += "*"
    if underscore_count % 2 != 0:
        text += "_"
        
    return text


def sanitize_output(text: str) -> str:
    """
    Sanitize and validate agent output before sending to the user.

    - Handles empty/None responses.
    - Truncates responses exceeding platform limits safely.
    - Attempts to fix broken Markdown caused by truncation.

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
        
        # Reserve space for the warning message
        warning_msg = "\n\n_(Resposta truncada por ser muito longa)_"
        max_allowed = TELEGRAM_MAX_LENGTH - len(warning_msg) - 10
        
        # Cut at max allowed
        truncated = cleaned[:max_allowed]
        
        # Try to find the last clean sentence boundary to avoid cutting mid-word
        last_period = truncated.rfind(".")
        last_newline = truncated.rfind("\n")
        
        # Prefer cutting at a newline if it's in the second half of the text, otherwise period
        cut_point = max(last_period, last_newline)
        
        if cut_point > max_allowed // 2:
            truncated = truncated[:cut_point + 1]

        # Prevent broken Markdown rendering in platforms like Telegram
        truncated = _balance_markdown(truncated)
        
        truncated += warning_msg
        return truncated

    return cleaned
