"""
Qdrant Cloud vector store — Hybrid Search (Dense + Sparse/BM25).

Manages the collection with dense and sparse vectors.
Search pipeline: Query Rewriting → Hybrid Search (K=8).
"""

import logging

from src.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    VECTOR_SEARCH_K,
)
from src.retrieval.embeddings import get_embeddings

logger = logging.getLogger(__name__)

# Gemini Embedding 2 Preview default dimension
EMBEDDING_DIMENSION = 3072 


def get_qdrant_client():
    """Initialize and return a Qdrant client."""
    from qdrant_client import QdrantClient
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL not found in environment variables.")
    logger.info(f"🔌 Connecting to Qdrant Cloud: {QDRANT_URL}")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def create_collection_if_needed(client) -> None:
    """Create the vector collection if it doesn't already exist, with Sparse support enabled."""
    from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams
    if client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(COLLECTION_NAME)
        logger.info(f"📦 Collection '{COLLECTION_NAME}' already exists ({info.points_count} points).")
        return

    logger.info(f"📦 Creating collection '{COLLECTION_NAME}' for Hybrid Search...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
        # Adding native support for sparse vectors (BM25)
        sparse_vectors_config={
            "langchain-sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        }
    )
    logger.info(f"✅ Collection '{COLLECTION_NAME}' created successfully with Sparse config.")


def init_cache_collection_if_needed(client) -> None:
    """Create the cache vector collection if it doesn't already exist."""
    from src.config import CACHE_COLLECTION_NAME
    from qdrant_client.models import Distance, VectorParams
    if client.collection_exists(CACHE_COLLECTION_NAME):
        return

    logger.info(f"📦 Creating cache collection '{CACHE_COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=CACHE_COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
    )
    logger.info(f"✅ Cache Collection '{CACHE_COLLECTION_NAME}' created successfully.")


def create_payload_indexes_if_needed(client) -> None:
    """Ensure that a Keyword Payload Index exists for metadata filtering."""
    from qdrant_client.models import PayloadSchemaType
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.source_filename",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info(f"✅ Payload Index enforced on '{COLLECTION_NAME}'.")
    except Exception as e:
        logger.warning(f"⚠️ Could not create payload index: {e}")


def get_vector_store():
    """Initialize the QdrantVectorStore with Hybrid Search (Dense + Sparse) enabled."""
    from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

    client = get_qdrant_client()
    create_collection_if_needed(client)
    create_payload_indexes_if_needed(client)

    # Dense embeddings (Gemini)
    embeddings = get_embeddings()
    
    # Sparse embeddings (BM25 local engine)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID, # Activates Dense + BM25 fusion
    )

    return vector_store


def get_retriever(k: int | None = None):
    """
    Get a LangChain retriever backed by Qdrant Hybrid Search.
    Internally uses Reciprocal Rank Fusion (RRF) to combine semantic and keyword scores.
    """
    search_k = k or VECTOR_SEARCH_K
    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={
            "k": search_k
        },
    )

    logger.info(f"🔍 Hybrid Retriever configured: k={search_k} (Dense + BM25)")
    return retriever
