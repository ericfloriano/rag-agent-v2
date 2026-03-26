"""
Document ingestion pipeline.

Loads PDFs and TXT files from docs/, splits into chunks,
enriches metadata, and upserts to Qdrant Cloud.

Usage:
    python -m src.retrieval.ingest
"""

import logging
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DOCUMENTS_DIR
from src.retrieval.vector_store import get_vector_store

logger = logging.getLogger(__name__)

# Chunking hyperparameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_documents(documents_dir: Path | None = None) -> list[Document]:
    """
    Load all PDF and TXT documents from the source directory.

    Args:
        documents_dir: Path to the documents directory. Defaults to DOCUMENTS_DIR.

    Returns:
        List of LangChain Documents (one per PDF page or TXT file).
    """
    docs_path = documents_dir or DOCUMENTS_DIR

    if not docs_path.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {docs_path}\n"
            f"Please place your PDF/TXT files in: {docs_path}"
        )

    pdf_files = sorted(docs_path.glob("*.pdf")) + sorted(docs_path.glob("*.PDF"))
    txt_files = sorted(docs_path.glob("*.txt"))

    total_files = len(pdf_files) + len(txt_files)
    if total_files == 0:
        raise FileNotFoundError(
            f"No PDF or TXT files found in: {docs_path}\n"
            f"Please add your ReCARE/RePAD documents to this directory."
        )

    logger.info(
        f"📂 Found {len(pdf_files)} PDF(s) and {len(txt_files)} TXT(s) "
        f"in {docs_path}"
    )

    all_documents: list[Document] = []

    # Load PDFs
    for pdf_file in pdf_files:
        logger.info(f"  📄 Loading PDF: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            all_documents.extend(docs)
            logger.info(f"     → {len(docs)} pages loaded")
        except Exception as e:
            logger.error(f"  ❌ Failed to load {pdf_file.name}: {e}")
            continue

    # Load TXT files
    for txt_file in txt_files:
        logger.info(f"  📝 Loading TXT: {txt_file.name}")
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs = loader.load()
            all_documents.extend(docs)
            logger.info(f"     → {len(docs)} document(s) loaded")
        except Exception as e:
            logger.error(f"  ❌ Failed to load {txt_file.name}: {e}")
            continue

    logger.info(f"📄 Total documents loaded: {len(all_documents)}")
    return all_documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into smaller chunks for embedding.

    Uses RecursiveCharacterTextSplitter with:
    - chunk_size: 1000 characters
    - chunk_overlap: 200 characters
    - Separators optimized for technical documents

    Args:
        documents: List of full-page documents.

    Returns:
        List of chunked documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    logger.info(
        f"✂️  Split {len(documents)} pages into {len(chunks)} chunks "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    return chunks


def enrich_metadata(chunks: list[Document]) -> list[Document]:
    """
    Enrich chunk metadata with useful fields for retrieval and debugging.

    Adds:
    - source_filename: basename of the original PDF
    - chunk_index: sequential index within the document

    Args:
        chunks: List of document chunks.

    Returns:
        Same list with enriched metadata.
    """
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        chunk.metadata["source_filename"] = Path(source).name
        chunk.metadata["chunk_index"] = i

    logger.info(f"🏷️  Enriched metadata for {len(chunks)} chunks")
    return chunks


def ingest_all(documents_dir: Path | None = None, batch_size: int = 20) -> int:
    """
    Full ingestion pipeline: Load → Split → Enrich → Upsert to Qdrant.

    Includes rate limiting to stay within Google Free Tier limits
    (100 embedding requests/minute for Gemini Embedding 2).

    Args:
        documents_dir: Path to the documents directory. Defaults to DOCUMENTS_DIR.
        batch_size: Number of documents to upsert per batch (default: 20).

    Returns:
        Total number of chunks ingested.
    """
    import time

    # Rate limit: 1500 requests/min (Gemini Free Tier)
    PAUSE_BETWEEN_BATCHES = 2  # seconds (reduced drastically to prevent Cloud Run timeouts)
    MAX_RETRIES = 3

    logger.info("🚀 Starting document ingestion pipeline...")

    # 1. Load documents
    documents = load_documents(documents_dir)

    # 2. Split into chunks
    chunks = split_documents(documents)

    # 3. Enrich metadata
    chunks = enrich_metadata(chunks)

    # 4. Get vector store and upsert
    vector_store = get_vector_store()

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    logger.info(
        f"📤 Upserting {len(chunks)} chunks to Qdrant "
        f"({total_batches} batches of {batch_size}, "
        f"{PAUSE_BETWEEN_BATCHES}s pause between batches)"
    )

    # Batch upsert with rate limiting
    total_upserted = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1

        # Retry logic for rate limit errors
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                vector_store.add_documents(batch)
                total_upserted += len(batch)
                logger.info(
                    f"   ✅ Batch {batch_num}/{total_batches}: "
                    f"{total_upserted}/{len(chunks)} chunks upserted"
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < MAX_RETRIES:
                    wait_time = PAUSE_BETWEEN_BATCHES * attempt
                    logger.warning(
                        f"   ⏳ Rate limited (attempt {attempt}/{MAX_RETRIES}). "
                        f"Waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise

        # Pause between batches to respect rate limits
        if i + batch_size < len(chunks):
            logger.info(f"   ⏳ Pausing {PAUSE_BETWEEN_BATCHES}s (rate limit)...")
            time.sleep(PAUSE_BETWEEN_BATCHES)

    logger.info(
        f"🎉 Ingestion complete! {total_upserted} chunks indexed in Qdrant."
    )
    return total_upserted


# === Standalone execution ===
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        total = ingest_all()
        print(f"\n✅ Successfully ingested {total} chunks into Qdrant Cloud.")
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        sys.exit(1)
