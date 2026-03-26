"""
Zero-Cost Semantic Caching using Qdrant Vector DB.

Stores previous questions and their LLM-generated answers.
If a new question is semantically identical (score > threshold) to a previous one,
returns the cached answer immediately, bypassing the LLM completely.
"""

import logging

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from src.retrieval.vector_store import get_qdrant_client, init_cache_collection_if_needed
from src.retrieval.embeddings import get_dense_embeddings
from src.config import CACHE_COLLECTION_NAME, SEMANTIC_CACHE_THRESHOLD

logger = logging.getLogger(__name__)

_cache_store: QdrantVectorStore | None = None


def get_cache_store() -> QdrantVectorStore:
    """Lazy initialize the Qdrant Vector Store for the cache collection."""
    global _cache_store
    if _cache_store is None:
        client = get_qdrant_client()
        init_cache_collection_if_needed(client)
        
        embeddings = get_dense_embeddings()
        _cache_store = QdrantVectorStore(
            client=client,
            collection_name=CACHE_COLLECTION_NAME,
            embedding=embeddings,
        )
    return _cache_store


async def check_cache(question: str) -> str | None:
    """
    Search the vector DB for a semantically identical question.
    
    Args:
        question: The user's input question.
        
    Returns:
        The cached generated response if similarity >= threshold, else None.
    """
    try:
        store = get_cache_store()
        
        # We need the score to evaluate semantic proximity
        results = await store.asimilarity_search_with_score(question, k=1)
        
        if not results:
            return None
            
        doc, score = results[0]
        
        logger.info(f"🧠 Semantic Cache check: Score = {score:.4f} (Threshold: {SEMANTIC_CACHE_THRESHOLD})")
        
        if score >= SEMANTIC_CACHE_THRESHOLD:
            logger.info("🌟 SEMANTIC CACHE HIT! Bypassing LLM generation.")
            return doc.metadata.get("response")
            
    except Exception as e:
        logger.warning(f"⚠️ Cache read failed (skipping cache): {e}")
        
    return None


async def add_to_cache(question: str, response: str) -> None:
    """
    Save a new question and its generated response to the vector cache.
    
    Args:
        question: The original user question.
        response: The final approved LLM generation.
    """
    try:
        store = get_cache_store()
        
        doc = Document(
            page_content=question,
            metadata={"response": response}
        )
        
        await store.aadd_documents([doc])
        logger.info("💾 Question & Answer saved to Semantic Cache.")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to write to cache: {e}")
