"""
Multi-LLM Cascade Router.
Implements the fallback strategy: Gemini → Groq → Kimi K2.5.
If the primary LLM fails (rate limit, timeout), it automatically
falls through to the next provider in the cascade.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from src.llm.providers import get_gemini_llm, get_groq_llm, get_kimi_llm

logger = logging.getLogger(__name__)


class LLMCascadeRouter:
    """
    Resilient Multi-LLM router with automatic fallback.

    Order: Gemini 2.0 Flash → Groq (LLaMA) → Kimi K2.5 (OpenRouter)

    Usage:
        router = LLMCascadeRouter()
        response = await router.ainvoke(messages)
    """

    def __init__(self):
        self._providers: list[tuple[str, BaseChatModel]] = []
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize LLM providers in cascade order. Skip unavailable ones."""
        provider_factories = [
            ("Gemini", get_gemini_llm),
            ("Groq", get_groq_llm),
            ("Kimi", get_kimi_llm),
        ]
        for name, factory in provider_factories:
            try:
                llm = factory()
                self._providers.append((name, llm))
                logger.info(f"✅ LLM Provider '{name}' loaded successfully.")
            except ValueError as e:
                logger.warning(f"⚠️ LLM Provider '{name}' skipped (missing API key): {e}")

        if not self._providers:
            raise RuntimeError("No LLM providers could be initialized. Check your API keys.")

        logger.info(
            f"🔗 LLM Cascade: {' → '.join(name for name, _ in self._providers)}"
        )

    @property
    def primary_llm(self) -> BaseChatModel:
        """Returns the first available LLM (for use in LangGraph nodes)."""
        return self._providers[0][1]

    async def ainvoke(self, messages: list[BaseMessage], **kwargs: Any) -> BaseMessage:
        """
        Invoke the LLM cascade asynchronously.
        Tries each provider in order until one succeeds.
        """
        last_error = None
        for name, llm in self._providers:
            try:
                logger.info(f"🤖 Attempting LLM invocation via '{name}'...")
                response = await llm.ainvoke(messages, **kwargs)
                logger.info(f"✅ LLM '{name}' responded successfully.")
                return response
            except Exception as e:
                logger.warning(f"⚠️ LLM '{name}' failed: {type(e).__name__}: {e}")
                last_error = e
                continue

        # All providers failed
        logger.error("❌ All LLM providers in the cascade have failed.")
        raise RuntimeError(
            f"All LLM providers failed. Last error: {last_error}"
        ) from last_error

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> BaseMessage:
        """
        Invoke the LLM cascade synchronously.
        Tries each provider in order until one succeeds.
        """
        last_error = None
        for name, llm in self._providers:
            try:
                logger.info(f"🤖 Attempting LLM invocation via '{name}'...")
                response = llm.invoke(messages, **kwargs)
                logger.info(f"✅ LLM '{name}' responded successfully.")
                return response
            except Exception as e:
                logger.warning(f"⚠️ LLM '{name}' failed: {type(e).__name__}: {e}")
                last_error = e
                continue

        logger.error("❌ All LLM providers in the cascade have failed.")
        raise RuntimeError(
            f"All LLM providers failed. Last error: {last_error}"
        ) from last_error
