"""
Embeddings module for Hybrid Search (Dense + Sparse).

Dense:  Google Gemini Embedding 001 (768 dimensions)
Sparse: FastEmbed BM25 for keyword matching
"""

import logging

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client.models import SparseVector

from src.config import GOOGLE_API_KEY, EMBEDDINGS_MODEL_NAME

logger = logging.getLogger(__name__)


def get_dense_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Initialize dense embeddings using Google Gemini Embedding 001.

    Returns:
        GoogleGenerativeAIEmbeddings instance (768 dimensions).
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found. Cannot initialize dense embeddings.")

    logger.info(f"🔢 Loading Dense Embeddings: {EMBEDDINGS_MODEL_NAME}")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDINGS_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_document",
    )


def get_query_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Initialize dense embeddings optimized for queries.

    Uses task_type='retrieval_query' for better query-document matching.

    Returns:
        GoogleGenerativeAIEmbeddings instance configured for queries.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found. Cannot initialize query embeddings.")

    logger.info(f"🔍 Loading Query Embeddings: {EMBEDDINGS_MODEL_NAME}")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDINGS_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_query",
    )
