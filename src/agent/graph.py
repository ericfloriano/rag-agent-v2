"""
LangGraph StateGraph definition.
Updated to support async event streaming.
"""

import logging
import json
from typing import AsyncGenerator

from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    classify_intent,
    retrieve,
    grade_documents,
    generate,
    # check_hallucination, # Deixado comentado conforme pedido
    generate_static_greeting,
    generate_static_out_of_scope,
    generate_static_fallback
)

logger = logging.getLogger(__name__)

# --- ROUTING AND DECISIONS ---
def route_intent(state: AgentState) -> str:
    intent = state.get("intent", "factual")
    if "greeting" in intent:
        return "generate_static_greeting"
    elif "out_of_scope" in intent:
        return "generate_static_out_of_scope"
    return "retrieve"

def decide_to_generate(state: AgentState) -> str:
    relevant_docs = state.get("relevant_documents", [])
    if not relevant_docs:
        return "generate_static_fallback"
    return "generate"

# --- FEATURE TOGGLE: Commented out to focus on UX speed ---
# def route_hallucination(state: AgentState) -> str:
#     is_hallucination = state.get("is_hallucination", False)
#     retry_count = state.get("retry_count", 0)
#     if is_hallucination and retry_count < 2:
#         logger.warning(f"🔄 Hallucination detected! Retrying generation (Attempt {retry_count + 1}/2)")
#         return "generate"
#     return END

# --- GRAPH CONSTRUCTION ---
def build_graph() -> StateGraph:
    logger.info("🏗️  Building Agentic LangGraph (Streaming Mode)...")

    graph = StateGraph(AgentState)

    # Adding Processing Nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    # graph.add_node("check_hallucination", check_hallucination) # DesativadoFeature Toggle
    
    # Static Response Nodes (Fallbacks)
    graph.add_node("generate_static_greeting", generate_static_greeting)
    graph.add_node("generate_static_out_of_scope", generate_static_out_of_scope)
    graph.add_node("generate_static_fallback", generate_static_fallback)

    # Entry Point Definition
    graph.set_entry_point("classify_intent")

    # Conditional Entry Flow (Intent Routing)
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "generate_static_greeting": "generate_static_greeting",
            "generate_static_out_of_scope": "generate_static_out_of_scope",
            "retrieve": "retrieve"
        }
    )
    
    # Static flows go directly to the END
    graph.add_edge("generate_static_greeting", END)
    graph.add_edge("generate_static_out_of_scope", END)
    graph.add_edge("generate_static_fallback", END)

    # Main RAG Flow
    graph.add_edge("retrieve", "grade_documents")
    
    graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate_static_fallback": "generate_static_fallback",
            "generate": "generate"
        }
    )

    # --- Previously went to check_hallucination. Now goes directly to the END to allow Streaming! ---
    graph.add_edge("generate", END)
    
    # graph.add_edge("generate", "check_hallucination")
    # graph.add_conditional_edges(
    #     "check_hallucination",
    #     route_hallucination,
    #     {
    #         "generate": "generate",
    #         END: END
    #     }
    # )

    compiled = graph.compile()
    logger.info("✅ Agentic LangGraph compiled successfully!")

    return compiled

# Graph Singleton
_compiled_graph = None

def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph

# ============================================================
# AGENT EXECUTION (STREAMING MODE)
# ============================================================
async def run_agent(question: str, chat_history: str = "") -> AsyncGenerator[str, None]:
    """
    Executes the LangGraph pipeline in streaming mode.
    Yields chunks of the final generated text as they arrive from the LLM.
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

    logger.info(f"🤖 Running agent in STREAMING mode for question: '{question[:80]}...'")
    
    try:        
        from langfuse.callback import CallbackHandler
        langfuse_handler = CallbackHandler()
                
        # Using astream_events (v1) to capture tokens in real-time
        # Filtering to catch only events arriving from the 'generate' node
        async for event in graph.astream_events(
            initial_state,
            version="v1",
            config={"callbacks": [langfuse_handler]}
        ):
            kind = event.get("event")
            node_name = event.get("metadata", {}).get("langgraph_node", "")

            # We only want the tokens generated by the final generation node
            if kind == "on_chat_model_stream" and node_name == "generate":
                content = event["data"]["chunk"].content
                if content:
                    # Submitting the captured token to Telegram
                    yield content
                    
    except Exception as e:
        # If a critical error happens in the flow, yield the static fallback message
        from src.agent.prompts import FALLBACK_ALL_FAILED
        logger.error(f"❌ LangGraph STREAMING pipeline failed: {type(e).__name__}: {e}", exc_info=True)
        # Yielding the entire fallback message at once since the flow broke down
        yield FALLBACK_ALL_FAILED
