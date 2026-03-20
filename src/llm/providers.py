"""
Multi-LLM providers initialization.
Configures each LLM provider (Gemini, Groq, Kimi) for the cascade router.
"""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src.config import (
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    OPENROUTER_API_KEY,
    GEMINI_MODEL_NAME,
    GROQ_MODEL_NAME,
    KIMI_MODEL_NAME,
)

logger = logging.getLogger(__name__)


def get_gemini_llm():
    """Primary LLM: Google Gemini 2.0 Flash."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    logger.info(f"Loading LLM: Gemini ({GEMINI_MODEL_NAME})")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY,
    )


def get_groq_llm():
    """Fallback 1: Groq (LLaMA / Mixtral via LPU)."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    logger.info(f"Loading LLM: Groq ({GROQ_MODEL_NAME})")
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=0.0,
        groq_api_key=GROQ_API_KEY,
    )


def get_kimi_llm():
    """Fallback 2: Moonshot Kimi K2.5 via OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables.")
    logger.info(f"Loading LLM: Kimi K2.5 ({KIMI_MODEL_NAME})")
    return ChatOpenAI(
        model=KIMI_MODEL_NAME,
        temperature=0.0,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
    )
