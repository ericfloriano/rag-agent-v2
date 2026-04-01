"""
Multi-LLM Intent Router.
Implements specific fallback strategies based on the task.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from src.llm.providers import get_gemini_llm, get_groq_llm, get_openrouter_llm, get_openai_llm
from src.config import (
    GEMINI_FLASH_MODEL,
    GROQ_FAST_MODEL,
    GROQ_REASON_MODEL,
    OPENROUTER_FAST_MODEL,
    OPENROUTER_REASON_MODEL,
    OPENROUTER_GEMMA_MODEL,
    OPENAI_MINI_MODEL # <--- Imported from config
)

logger = logging.getLogger(__name__)

class LLMIntentRouter:

    def __init__(self, task_type: str = "default"):
        self.task_type = task_type
        self._providers: list[tuple[str, BaseChatModel]] = []
        self._initialize_providers()

    def _initialize_providers(self):
        
        # 1. Classify / Rewrite -> Ultra Fast (Groq leading)
        if self.task_type in ["classify_intent", "rewrite_query"]:
            route_plan = [
                ("Groq (Llama 3.1 8B)", get_groq_llm, GROQ_FAST_MODEL),
                ("OpenAI (Mini)", get_openai_llm, OPENAI_MINI_MODEL),
            ]
        # 2. Grade Documents -> Fast, structured parallel reasoning (OpenAI leading)
        elif self.task_type == "grade_documents":
            route_plan = [
                ("OpenAI (Mini)", get_openai_llm, OPENAI_MINI_MODEL),
                ("Groq (Llama 3.3 70B)", get_groq_llm, GROQ_REASON_MODEL),
            ]
        # 3. Generate -> Factuality, synthesis, and strict formatting (OpenAI leading)
        elif self.task_type == "generate_factual":
            route_plan = [
                ("OpenAI (Mini)", get_openai_llm, OPENAI_MINI_MODEL),
                ("Groq (Llama 3.3 70B)", get_groq_llm, GROQ_REASON_MODEL),
            ]
        else:
            route_plan = [
                ("OpenAI (Mini)", get_openai_llm, OPENAI_MINI_MODEL),
                ("Groq (Fast)", get_groq_llm, GROQ_FAST_MODEL),
            ]

        for name, factory, model_name in route_plan:
            try:
                llm = factory(model_name)
                self._providers.append((name, llm))
                logger.info(f"✅ LLM Provider '{name}' loaded successfully for '{self.task_type}'.")
            except ValueError as e:
                logger.warning(f"⚠️ LLM Provider '{name}' skipped: {e}")

        if not self._providers:
            raise RuntimeError(f"No LLM providers could be initialized for task '{self.task_type}'.")

        logger.info(
            f"🔗 LLM Cascade for [{self.task_type}]: {' → '.join(name for name, _ in self._providers)}"
        )

    @property
    def primary_llm(self) -> BaseChatModel:
        return self._providers[0][1]

    async def ainvoke(self, messages: list[BaseMessage], **kwargs: Any) -> BaseMessage:
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

        logger.error(f"❌ [{self.task_type}] All LLM providers in the cascade have failed.")
        raise RuntimeError(f"All LLM providers failed for '{self.task_type}'.") from last_error

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> BaseMessage:
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
        raise RuntimeError(f"All LLM providers failed for '{self.task_type}'.") from last_error
