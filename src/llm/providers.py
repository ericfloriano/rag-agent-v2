"""
Multi-LLM providers initialization.
Configures each LLM provider (Gemini, Groq, Kimi) for the cascade router.
"""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI
import langchain_google_genai.chat_models
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from typing import Callable, Any

# --- MONKEY PATCH PARA FAST FAIL (Evitar atraso no Rate Limit do Gemini) ---
# A biblioteca langchain_google_genai usa `tenacity` internamente para tentar
# chamadas repetidas quando há limite de cota (429), ignorando `max_retries=0`.
# Nós substituímos os decoradores por chamadas diretas para garantir fallback
# instantâneo para o Groq LLaMA ao esgotar a cota gratuita do Gemini.
async def _achat_fast_fail(generation_method: Callable, **kwargs: Any) -> Any:
    try:
        return await generation_method(**kwargs)
    except Exception as e:
        raise e

def _chat_fast_fail(generation_method: Callable, **kwargs: Any) -> Any:
    try:
        return generation_method(**kwargs)
    except Exception as e:
        raise e

# Aplicando os patches globais
langchain_google_genai.chat_models._achat_with_retry = _achat_fast_fail
langchain_google_genai.chat_models._chat_with_retry = _chat_fast_fail
# -------------------------------------------------------------------------

from src.config import (
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    OPENROUTER_API_KEY,
)

logger = logging.getLogger(__name__)


def get_gemini_llm(model_name: str):
    """LLM: Google Gemini via langchain-google-genai."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    logger.info(f"Loading LLM: Gemini ({model_name})")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY,
        max_retries=0,  # Fast fail para acionar o LLM Router imediatamente
    )


def get_groq_llm(model_name: str):
    """LLM: Groq via langchain-groq."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    logger.info(f"Loading LLM: Groq ({model_name})")
    return ChatGroq(
        model=model_name,
        temperature=0.0,
        groq_api_key=GROQ_API_KEY,
        max_retries=0,
    )


def get_openrouter_llm(model_name: str):
    """LLM: OpenRouter via langchain-openai."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables.")
    logger.info(f"Loading LLM: OpenRouter ({model_name})")
    return ChatOpenAI(
        model=model_name,
        temperature=0.0,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        max_retries=0,
    )
