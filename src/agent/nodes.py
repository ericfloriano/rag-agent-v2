"""
LangGraph agent nodes.

Implements all the logic blocks for the Agentic RAG:
Classification, Retrieval (Hybrid Search), Parallel Grading, and Generation.
"""

import logging
import asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from src.agent.state import AgentState

logger = logging.getLogger(__name__)

# Module-level singleton (initialized on first use)
_retriever = None

def _get_retriever():
    """Lazy loading for the dense retriever."""
    from src.retrieval.vector_store import get_retriever
    from src.config import VECTOR_SEARCH_K
    global _retriever
    if _retriever is None:
        _retriever = get_retriever(k=VECTOR_SEARCH_K)
    return _retriever


# ============================================================
# NODES
# ============================================================

async def classify_intent(state: AgentState) -> dict:
    """Classify user intent to route properly (Greeting, Out of Scope, or Factual)."""
    from src.agent.prompts import CLASSIFY_INTENT_SYSTEM_PROMPT
    from src.llm.router import LLMIntentRouter
    
    question = state["question"]
    router = LLMIntentRouter(task_type="classify_intent")
    
    response = await router.ainvoke([
        SystemMessage(content=CLASSIFY_INTENT_SYSTEM_PROMPT),
        HumanMessage(content=question)
    ])
    
    intent = response.content.strip().lower()
    
    # Safe parsing to ensure strict routing
    if "<greeting>" in intent or "greeting" in intent:
        final_intent = "greeting"
    elif "<out_of_scope>" in intent or "out_of_scope" in intent:
        final_intent = "out_of_scope"
    else:
        final_intent = "factual"
        
    logger.info(f"🧠 Intent classified as: {final_intent}")
    return {"intent": final_intent}


async def retrieve(state: AgentState) -> dict:
    """Rewrite the query using chat history and retrieve documents via Dense Search."""
    from src.agent.rewriter import rewrite_query
    
    search_query = rewrite_query(state["question"], state.get("chat_history", ""))
    logger.info(f"🔍 Retrieving documents for: '{search_query[:80]}...'")

    retriever = _get_retriever()
    documents = await retriever.ainvoke(search_query)

    return {"documents": documents}


async def grade_documents(state: AgentState) -> dict:
    """Filter out documents that are not relevant to the question using parallel async requests."""
    from src.agent.prompts import GRADE_DOCUMENT_SYSTEM_PROMPT, GRADE_DOCUMENT_HUMAN_PROMPT
    from src.llm.router import LLMIntentRouter
    
    question = state["question"]
    documents = state.get("documents", [])
    router = LLMIntentRouter(task_type="grade_documents")
    
    # Helper function to process a single document asynchronously
    async def _grade_single_doc(doc):
        prompt = GRADE_DOCUMENT_HUMAN_PROMPT.format(document_content=doc.page_content, question=question)
        try:
            response = await router.ainvoke([
                SystemMessage(content=GRADE_DOCUMENT_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])
            # If the LLM says "sim" (yes), keep the document
            if "sim" in response.content.lower():
                return doc
        except Exception as e:
            logger.error(f"Error grading document: {e}")
        return None

    # Parallel execution: Run all evaluations concurrently
    tasks = [_grade_single_doc(doc) for doc in documents]
    results = await asyncio.gather(*tasks)
    
    # Filter out the failed/irrelevant documents (None values)
    relevant_docs = [doc for doc in results if doc is not None]
            
    logger.info(f"📑 Grader kept {len(relevant_docs)} out of {len(documents)} documents.")
    return {"relevant_documents": relevant_docs}


async def generate(state: AgentState) -> dict:
    """Generate the final answer based on the relevant documents context."""
    from src.agent.prompts import GENERATE_ANSWER_SYSTEM_PROMPT, GENERATE_ANSWER_HUMAN_PROMPT
    from src.llm.router import LLMIntentRouter

    question = state["question"]
    relevant_docs = state.get("relevant_documents", [])
    retry_count = state.get("retry_count", 0)

    context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)
    router = LLMIntentRouter(task_type="generate_factual")
    
    response = await router.ainvoke([
        SystemMessage(content=GENERATE_ANSWER_SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=GENERATE_ANSWER_HUMAN_PROMPT.format(question=question))
    ])

    return {
        "generation": response.content.strip(),
        "llm_provider_used": response.response_metadata.get("model_name", "Unknown Model"),
        "retry_count": retry_count + 1 # Increment for hallucination loop
    }


async def check_hallucination(state: AgentState) -> dict:
    """Check if the generation is grounded in the documents (Self-Correction Loop)."""
    from src.agent.prompts import HALLUCINATION_CHECK_SYSTEM_PROMPT, HALLUCINATION_CHECK_HUMAN_PROMPT
    from src.llm.router import LLMIntentRouter
    
    generation = state["generation"]
    relevant_docs = state.get("relevant_documents", [])
    context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)
    
    router = LLMIntentRouter(task_type="grade_documents") # Uses same router tier as grading
    response = await router.ainvoke([
        SystemMessage(content=HALLUCINATION_CHECK_SYSTEM_PROMPT),
        HumanMessage(content=HALLUCINATION_CHECK_HUMAN_PROMPT.format(context=context, generation=generation))
    ])
    
    is_hallucination = "alucinacao" in response.content.lower() or "alucinação" in response.content.lower()
    return {"is_hallucination": is_hallucination}


# ============================================================
# STATIC RESPONSES NODES
# ============================================================

async def generate_static_greeting(state: AgentState) -> dict:
    from src.agent.prompts import GREETING_RESPONSE
    return {"generation": GREETING_RESPONSE}

async def generate_static_out_of_scope(state: AgentState) -> dict:
    from src.agent.prompts import FALLBACK_OUT_OF_SCOPE
    return {"generation": FALLBACK_OUT_OF_SCOPE}

async def generate_static_fallback(state: AgentState) -> dict:
    from src.agent.prompts import FALLBACK_NO_RELEVANT_DOCS
    return {"generation": FALLBACK_NO_RELEVANT_DOCS}
