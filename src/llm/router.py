"""
Multi-LLM Intent Router.
Implements specific fallback strategies based on the task (classification, grading, generation).
If the primary LLM fails (rate limit, timeout), it automatically
falls through to the next provider in the specific cascade.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from src.llm.providers import get_gemini_llm, get_groq_llm, get_openrouter_llm
from src.config import (
    GEMINI_FLASH_MODEL,
    GROQ_FAST_MODEL,
    GROQ_REASON_MODEL,
    OPENROUTER_FAST_MODEL,
    OPENROUTER_REASON_MODEL,
    OPENROUTER_GEMMA_MODEL
)

logger = logging.getLogger(__name__)


class LLMIntentRouter:
    """
    Resilient Multi-LLM router with intent-based automatic fallback.

    Usage:
        router = LLMIntentRouter(task_type="classify_intent")
        response = await router.ainvoke(messages)
    """

    def __init__(self, task_type: str = "default"):
        self.task_type = task_type
        self._providers: list[tuple[str, BaseChatModel]] = []
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize LLM providers based on the task type intent."""
        
        # Define routes based on task_type
        # format: list of tuples (Provider Name, Factory Function, Model Name)
        if self.task_type in ["classify_intent", "rewrite_query"]:
            # Needs speed, low reasoning
            route_plan = [
                ("Groq (Fast)", get_groq_llm, GROQ_FAST_MODEL),
                ("OpenRouter (Fast)", get_openrouter_llm, OPENROUTER_FAST_MODEL),
                ("Gemini", get_gemini_llm, GEMINI_FLASH_MODEL)
            ]
        elif self.task_type == "grade_documents":
            # Needs precision, so Groq first, then Gemini (large context)
            route_plan = [
                ("Groq (Reasoning)", get_groq_llm, GROQ_REASON_MODEL),
                ("OpenRouter (Gemma)", get_openrouter_llm, OPENROUTER_GEMMA_MODEL),
                ("Gemini", get_gemini_llm, GEMINI_FLASH_MODEL)
            ]
        elif self.task_type == "generate_factual":
            # Needs complex reasoning and RAG extraction
            route_plan = [
                ("Groq (Reasoning)", get_groq_llm, GROQ_REASON_MODEL),
                ("OpenRouter (Reasoning)", get_openrouter_llm, OPENROUTER_REASON_MODEL),
                ("Gemini", get_gemini_llm, GEMINI_FLASH_MODEL)
            ]
        elif self.task_type == "generate_coaching":
            # Needs fluency and empathy
            route_plan = [
                ("Groq (Reasoning)", get_groq_llm, GROQ_REASON_MODEL),
                ("OpenRouter (Fast)", get_openrouter_llm, OPENROUTER_FAST_MODEL),
                ("Gemini", get_gemini_llm, GEMINI_FLASH_MODEL)
            ]
        else:
            # Default
            route_plan = [
                ("Groq (Reasoning)", get_groq_llm, GROQ_REASON_MODEL),
                ("OpenRouter (Fast)", get_openrouter_llm, OPENROUTER_FAST_MODEL),
                ("Gemini", get_gemini_llm, GEMINI_FLASH_MODEL)
            ]

        for name, factory, model_name in route_plan:
            try:
                llm = factory(model_name)
                self._providers.append((name, llm))
                logger.info(f"✅ LLM Provider '{name}' loaded successfully for '{self.task_type}'.")
            except ValueError as e:
                logger.warning(f"⚠️ LLM Provider '{name}' skipped (missing API key): {e}")

        if not self._providers:
            raise RuntimeError(f"No LLM providers could be initialized for task '{self.task_type}'.")

        logger.info(
            f"🔗 LLM Cascade for [{self.task_type}]: {' → '.join(name for name, _ in self._providers)}"
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
                logger.info(f"🤖 [{self.task_type}] Attempting LLM invocation via '{name}'...")
                response = await llm.ainvoke(messages, **kwargs)
                logger.info(f"✅ [{self.task_type}] LLM '{name}' responded successfully.")
                return response
            except Exception as e:
                logger.warning(f"⚠️ [{self.task_type}] LLM '{name}' failed: {type(e).__name__}: {e}")
                last_error = e
                continue

        # All providers failed
        logger.error(f"❌ [{self.task_type}] All LLM providers in the cascade have failed.")
        raise RuntimeError(
            f"All LLM providers failed for '{self.task_type}'. Last error: {last_error}"
        ) from last_error

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> BaseMessage:
        """
        Invoke the LLM cascade synchronously.
        """
        last_error = None
        for name, llm in self._providers:
            try:
                logger.info(f"🤖 [{self.task_type}] Attempting LLM invocation via '{name}'...")
                response = llm.invoke(messages, **kwargs)
                logger.info(f"✅ [{self.task_type}] LLM '{name}' responded successfully.")
                return response
            except Exception as e:
                logger.warning(f"⚠️ [{self.task_type}] LLM '{name}' failed: {type(e).__name__}: {e}")
                last_error = e
                continue

        logger.error(f"❌ [{self.task_type}] All LLM providers in the cascade have failed.")
        raise RuntimeError(
            f"All LLM providers failed for '{self.task_type}'. Last error: {last_error}"
        ) from last_error
