"""
LangGraph agent nodes.

Simplified Architecture: Retrieve (Top 5) -> Generate (Single LLM Call)
"""

import logging
from langchain_core.messages import HumanMessage

from src.agent.prompts import GENERATE_ANSWER_PROMPT
from src.agent.state import AgentState
from src.llm.router import LLMIntentRouter
from src.retrieval.vector_store import get_retriever

logger = logging.getLogger(__name__)

# Module-level singletons (initialized on first use)
_retriever = None

def _get_retriever():
    """Lazy initialization of the retriever."""
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


# ============================================================
# NODE: retrieve
# ============================================================

async def retrieve(state: AgentState) -> dict:
    """
    Retrieve relevant documents from Qdrant using hybrid search.
    Always returns top 5 to keep the prompt context lean and avoid limits.
    """
    question = state["question"]
    logger.info(f"🔍 Retrieving documents for: '{question[:80]}...'")

    retriever = _get_retriever()
    documents = await retriever.ainvoke(question)
    
    # Anti Rate-Limit / Context Bloat: Limitamos ao Top 5
    documents = documents[:5]

    logger.info(f"📄 Retrieved {len(documents)} context chunks for generation.")
    return {"documents": documents}


# ============================================================
# NODE: generate
# ============================================================

async def generate(state: AgentState) -> dict:
    """
    Generate an answer using relevant documents as context.
    The LLM Prompt itself decides if it's a greeting, factual, or out_of_scope.
    This saves 3 API calls per user message.
    """
    question = state["question"]
    documents = state.get("documents", [])
    chat_history = state.get("chat_history", "")

    # Build context from documents
    context = ""
    if documents:
        context = "\n\n---\n\n".join(
            f"[Fonte: {doc.metadata.get('source_filename', 'N/A')}, "
            f"Página: {doc.metadata.get('page', 'N/A')}]\n"
            f"{doc.page_content}"
            for doc in documents
        )
    else:
        context = "NENHUM DOCUMENTO ENCONTRADO NO BANCO DE DADOS (RAG VAZIO)."

    logger.info(f"✍️  Generating AI answer with single Mega-Prompt...")

    router = LLMIntentRouter(task_type="generate_factual")
    prompt = GENERATE_ANSWER_PROMPT.format(
        chat_history=chat_history or "(sem histórico)",
        context=context,
        question=question,
    )
    
    response = await router.ainvoke([HumanMessage(content=prompt)])
    generation = response.content.strip()

    # Get model name for logging
    model_info = response.response_metadata.get("model_name", "Unknown Model")
    logger.info(f"✅ AI Answer generated using {model_info}")

    return {
        "generation": generation,
        "llm_provider_used": model_info,
    }
