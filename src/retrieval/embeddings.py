"""
Embeddings module — Dense Semantic Search.

Uses Google Gemini Embedding for high-quality semantic representation.
LangChain natively handles the switch between 'retrieval_document' (indexing)
and 'retrieval_query' (searching) automatically under the hood.
"""

import logging

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import GOOGLE_API_KEY, EMBEDDINGS_MODEL_NAME

logger = logging.getLogger(__name__)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Initialize dense embeddings using Google Gemini.

    Returns:
        GoogleGenerativeAIEmbeddings instance.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found. Cannot initialize dense embeddings.")

    logger.info(f"🔢 Loading Embeddings: {EMBEDDINGS_MODEL_NAME}")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDINGS_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        # We don't hardcode task_type here because LangChain handles it dynamically.
    )

# Mantendo os aliases por compatibilidade caso sejam usados em outras partes do código
get_dense_embeddings = get_embeddings
get_query_embeddings = get_embeddings
