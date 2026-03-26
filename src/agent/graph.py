"""
LangGraph StateGraph definition.

Wires all agent nodes together with conditional edges to create
the Agentic RAG pipeline with self-correction loops.

Simplified Graph flow (Cost Optimized):
    START → retrieve → generate → END
"""

import logging

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import retrieve, generate

logger = logging.getLogger(__name__)


# ============================================================
# Graph Builder
# ============================================================

def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph agent.

    Returns:
        Compiled StateGraph ready to be invoked.
    """
    logger.info("🏗️  Building LangGraph agent...")

    graph = StateGraph(AgentState)

    # --- Add nodes ---
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    # --- Set entry point ---
    graph.set_entry_point("retrieve")

    # --- Edges ---
    # After retrieve → always generate
    graph.add_edge("retrieve", "generate")

    # Terminal nodes → END
    graph.add_edge("generate", END)

    # --- Compile ---
    compiled = graph.compile()
    logger.info("✅ LangGraph agent compiled successfully!")

    return compiled


# Module-level compiled graph (lazy singleton)
_compiled_graph = None


def get_graph():
    """
    Get the compiled LangGraph agent (singleton).

    Returns:
        Compiled StateGraph.
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def run_agent(question: str, chat_history: str = "") -> str:
    """
    Convenience function to run the agent with a question.

    Args:
        question: User's question (text or transcribed from audio).
        chat_history: Formatted conversation history (optional).

    Returns:
        The agent's response string.
    """
    graph = get_graph()

    initial_state: AgentState = {
        "question": question,
        "chat_history": chat_history,
        "intent": "",
        "documents": [],
        "relevant_documents": [],
        "generation": "",
        "is_hallucination": False,
        "retry_count": 0,
        "llm_provider_used": "",
    }

    logger.info(f"🤖 Running agent for question: '{question[:80]}...'")
    
    try:
        result = await graph.ainvoke(initial_state)
        response = result.get("generation", "Erro interno: nenhuma resposta gerada.")
    except Exception as e:
        logger.error(f"❌ Grafo do LangGraph falhou. Provedores esgotados: {e}")
        response = "Desculpe, nossos sistemas estão temporariamente sobrecarregados. Por favor, aguarde alguns instantes e tente novamente."
        
    logger.info(f"🤖 Agent response: '{response[:80]}...'")

    return response
