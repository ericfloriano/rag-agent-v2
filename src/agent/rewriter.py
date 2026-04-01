"""
Query Rewriter Module.
Handles the adaptation of user queries based on conversation history.
"""
import os
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

def rewrite_query(query: str, chat_history: str = "") -> str:
    """
    Analyzes the conversation history and the new query.
    If the question requires contextual resolution, it leverages the LLM to rewrite it.
    """
    if not chat_history or chat_history.strip() == "":
        logger.info(f"🔄 No chat history available. Keeping original query.")
        return query
        
    logger.info(f"🔄 Rewriting query based on history: '{query[:50]}...'")
    
    try:
        from src.agent.prompts import REWRITER_SYSTEM_PROMPT, REWRITER_HUMAN_PROMPT
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", REWRITER_SYSTEM_PROMPT),
            ("human", REWRITER_HUMAN_PROMPT)
        ])
                
        llm = ChatGroq(
            temperature=0, 
            model_name=os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant"),
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        chain = prompt | llm
        response = chain.invoke({"chat_history": chat_history, "query": query})
        
        rewritten = response.content.strip()
        logger.info(f"✨ Query rewritten for Qdrant: '{rewritten}'")
        return rewritten
        
    except Exception as e:
        logger.error(f"⚠️ Failed to rewrite query, falling back to original: {e}")
        return query
