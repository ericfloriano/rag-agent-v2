"""
Agent state definition for the LangGraph agent.

Defines the TypedDict that flows through every node in the graph.
All nodes read from and write to this shared state.
"""

from typing import TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    """
    State that flows through the LangGraph agent.

    Attributes:
        question: User's question (text or transcribed from audio).
        chat_history: Formatted conversation history from Firestore.
        intent: Classified intent — "greeting", "factual", or "out_of_scope".
        documents: Raw documents retrieved from Qdrant.
        relevant_documents: Documents that passed relevance grading.
        generation: Generated response from the LLM.
        is_hallucination: Whether the generation failed the hallucination check.
        retry_count: Number of retrieval retries attempted (max: MAX_RETRIES).
        llm_provider_used: Which LLM provider generated the response (for logging).
    """

    question: str
    chat_history: str
    intent: str
    documents: list[Document]
    relevant_documents: list[Document]
    generation: str
    is_hallucination: bool
    retry_count: int
    llm_provider_used: str
