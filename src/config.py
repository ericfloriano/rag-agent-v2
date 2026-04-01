"""
Central configuration module.
Loads environment variables and sets robust defaults for the application.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "docs"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash")
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.3-70b-versatile")
GROQ_REASON_MODEL = os.getenv("GROQ_REASON_MODEL", "llama-3.3-70b-versatile")
OPENROUTER_FAST_MODEL = os.getenv("OPENROUTER_FAST_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
OPENROUTER_REASON_MODEL = os.getenv("OPENROUTER_REASON_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
OPENROUTER_GEMMA_MODEL = os.getenv("OPENROUTER_GEMMA_MODEL", "google/gemma-2-9b-it:free")
OPENAI_MINI_MODEL = os.getenv("OPENAI_MINI_MODEL", "gpt-4o-mini")

EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME", "models/gemini-embedding-2-preview")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "whisper-large-v3-turbo")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "visuri_knowledge_base")
CACHE_COLLECTION_NAME = os.getenv("CACHE_COLLECTION_NAME", "visuri_semantic_cache")

VECTOR_SEARCH_K = int(os.getenv("VECTOR_SEARCH_K", "20"))
FINAL_DOCS_K = int(os.getenv("FINAL_DOCS_K", "8"))
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95"))

USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "ms-marco-MiniLM-L-12-v2")
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "5"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
USE_WEBHOOK = os.getenv("USE_WEBHOOK", "false").lower() == "true"
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "chat_sessions")

ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "change-me-in-production")
TURNSTILE_SITEKEY = os.getenv("TURNSTILE_SITEKEY", "")
TURNSTILE_SECRET = os.getenv("TURNSTILE_SECRET", "")

LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-agent-v2")
