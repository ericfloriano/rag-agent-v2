"""
Document ingestion pipeline.

Loads PDFs and TXT files, splits into chunks, enriches metadata, 
and upserts to Qdrant Cloud safely with AUTO-RESUME capability.
"""

import logging
import sys
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DOCUMENTS_DIR
from src.retrieval.vector_store import get_vector_store

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300


def load_documents(documents_dir: Path | None = None) -> list[Document]:
    docs_path = documents_dir or DOCUMENTS_DIR

    if not docs_path.exists():
        logger.warning(f"Documents directory not found: {docs_path}")
        return []

    pdf_files = sorted(docs_path.glob("*.pdf")) + sorted(docs_path.glob("*.PDF"))
    txt_files = sorted(docs_path.glob("*.txt"))

    all_documents: list[Document] = []

    for pdf_file in pdf_files:
        logger.info(f"  📄 Loading PDF: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            all_documents.extend(loader.load())
        except Exception as e:
            logger.error(f"  ❌ Failed to load {pdf_file.name}: {e}")

    for txt_file in txt_files:
        logger.info(f"  📝 Loading TXT: {txt_file.name}")
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            all_documents.extend(loader.load())
        except Exception as e:
            logger.error(f"  ❌ Failed to load {txt_file.name}: {e}")

    return all_documents


def split_documents(documents: list[Document]) -> list[Document]:
    if not documents:
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"✂️  Split {len(documents)} pages into {len(chunks)} chunks")
    return chunks


def enrich_metadata(chunks: list[Document]) -> list[Document]:
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        chunk.metadata["source_filename"] = Path(source).name
        chunk.metadata["chunk_index"] = i
    return chunks


def ingest_all(documents_dir: Path | None = None, batch_size: int = 10, reset: bool = False) -> int:
    """
    Full ingestion pipeline: Load → Split → Enrich → Upsert to Qdrant.
    Features 'Turtle Mode' and Auto-Resume for rate limits and connection drops.
    """
    logger.info("🚀 Starting document ingestion pipeline (Turtle Mode 🐢)...")

    documents = load_documents(documents_dir)
    if not documents:
        logger.info("No documents to ingest.")
        return 0

    chunks = split_documents(documents)
    chunks = enrich_metadata(chunks)

    from src.retrieval.vector_store import get_qdrant_client, COLLECTION_NAME
    from src.retrieval.vector_store import create_collection_if_needed, create_payload_indexes_if_needed
    
    client = get_qdrant_client()
    
    # Intelligence Logic: Reset or Resume?
    start_index = 0
    if reset and client.collection_exists(COLLECTION_NAME):
        logger.warning(f"🗑️  Dropping old collection '{COLLECTION_NAME}' (Reset=True)...")
        client.delete_collection(COLLECTION_NAME)
    elif client.collection_exists(COLLECTION_NAME):
        # Discover how many documents are already saved in Qdrant
        info = client.get_collection(COLLECTION_NAME)
        start_index = info.points_count
        if start_index > 0:
            logger.info(f"🔄 Auto-Resume activated: Found {start_index} chunks in DB. Skipping already sent ones...")

    create_collection_if_needed(client)
    create_payload_indexes_if_needed(client)
    
    vector_store = get_vector_store()
    
    # If the DB already has more chunks than our PDFs, something is wrong (deleted files?)
    if start_index >= len(chunks):
        logger.info("✅ All documents are already in the DB! No action needed.")
        return len(chunks)

    total_upserted = start_index 
    logger.info(f"📤 Preparing to upsert remaining {len(chunks) - start_index} chunks in batches of {batch_size}...")

    # Turtle Loop (Starts from dynamic start_index)
    i = start_index
    while i < len(chunks):
        batch = chunks[i : i + batch_size]
        try:
            vector_store.add_documents(batch)
            total_upserted += len(batch)
            logger.info(f"   ✅ {total_upserted}/{len(chunks)} chunks upserted")
            
            i += batch_size 
            
            # Micro-pause
            time.sleep(6) 
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"   ❌ Batch failed: {e}")
            
            # Shield against drops and Google Timeout
            if "429" in error_msg or "quota" in error_msg or "504" in error_msg or "deadline" in error_msg:
                logger.warning("   🛑 Google API choked! Sleeping for 65 seconds to try the same batch again...")
                time.sleep(65)
                # 'i' doesn't increment, so it tries again in the next loop
            else:
                logger.error("   ❌ Critical error. Aborting to avoid infinite loop.")
                raise e

    logger.info(f"🎉 Ingestion complete! {total_upserted} chunks indexed.")
    return total_upserted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        # reset=False so it can resume from where it left off.
        # In the future, if you want to delete everything and start from scratch, just change to reset=True.
        total = ingest_all(reset=False) 
        print(f"\n✅ Successfully ingested {total} chunks.")
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        sys.exit(1)
