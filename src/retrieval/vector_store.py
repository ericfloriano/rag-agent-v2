"""
Qdrant Cloud vector store connection and hybrid retrieval.

Manages the collection 'recare_knowledge_base_v2' with dense vectors
(Gemini Embedding 2 Preview, 3072 dimensions).
"""

import logging
from functools import lru_cache

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    VECTOR_SEARCH_K,
)
from src.retrieval.embeddings import get_dense_embeddings

logger = logging.getLogger(__name__)

# Gemini Embedding 2 Preview default dimension
EMBEDDING_DIMENSION = 3072


def get_qdrant_client() -> QdrantClient:
    """
    Initialize the Qdrant Cloud client.

    Returns:
        QdrantClient connected to the remote Qdrant Cloud instance.
    """
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError(
            "QDRANT_URL and QDRANT_API_KEY must be set in environment variables."
        )

    logger.info(f"🔌 Connecting to Qdrant Cloud: {QDRANT_URL}")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    logger.info("✅ Connected to Qdrant Cloud successfully.")
    return client


def create_collection_if_needed(client: QdrantClient) -> None:
    """
    Create the vector collection if it doesn't already exist.

    Configures:
    - Dense vectors: 768 dimensions (Gemini Embedding 001), Cosine similarity
    """
    if client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(COLLECTION_NAME)
        logger.info(
            f"📦 Collection '{COLLECTION_NAME}' already exists "
            f"({info.points_count} points)."
        )
        return

    logger.info(f"📦 Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=Distance.COSINE,
        ),
    )
    logger.info(f"✅ Collection '{COLLECTION_NAME}' created successfully.")


def init_cache_collection_if_needed(client: QdrantClient) -> None:
    """
    Create the cache vector collection if it doesn't already exist.
    Stores previous User Questions and Assistant Responses.
    """
    from src.config import CACHE_COLLECTION_NAME
    if client.collection_exists(CACHE_COLLECTION_NAME):
        info = client.get_collection(CACHE_COLLECTION_NAME)
        logger.info(f"📦 Cache Collection '{CACHE_COLLECTION_NAME}' already exists ({info.points_count} items).")
        return

    logger.info(f"📦 Creating cache collection '{CACHE_COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=CACHE_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=Distance.COSINE,
        ),
    )
    logger.info(f"✅ Cache Collection '{CACHE_COLLECTION_NAME}' created successfully.")


def create_payload_indexes_if_needed(client: QdrantClient) -> None:
    """
    Ensure that a Keyword Payload Index exists for 'metadata.source_filename'.
    Required by Qdrant to perform fast filtered deletions.
    """
    from qdrant_client.models import PayloadSchemaType
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.source_filename",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info(f"✅ Payload Index 'metadata.source_filename' enforced on '{COLLECTION_NAME}'.")
    except Exception as e:
        logger.warning(f"⚠️ Could not create payload index: {e}")


def get_vector_store() -> QdrantVectorStore:
    """
    Initialize the QdrantVectorStore connected to Qdrant Cloud.

    Uses dense embeddings (Gemini Embedding 001) for similarity search.

    Returns:
        QdrantVectorStore instance ready for similarity search and document insertion.
    """
    client = get_qdrant_client()
    create_collection_if_needed(client)
    create_payload_indexes_if_needed(client)

    embeddings = get_dense_embeddings()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    logger.info(
        f"🔍 VectorStore ready: collection='{COLLECTION_NAME}', "
        f"search_k={VECTOR_SEARCH_K}"
    )
    return vector_store


def get_retriever(search_k: int | None = None):
    """
    Get a LangChain retriever backed by Qdrant hybrid search.

    Args:
        search_k: Number of documents to retrieve. Defaults to VECTOR_SEARCH_K (10).

    Returns:
        A LangChain retriever for use in the RAG pipeline.
    """
    k = search_k or VECTOR_SEARCH_K
    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    logger.info(f"🔍 Retriever configured: k={k}")
    return retriever
