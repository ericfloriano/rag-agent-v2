"""
Centralized configuration for the Agentic RAG v2 application.
All environment variables and hyperparameters are managed here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documentos_fonte"

# === LLM Providers ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# === LLM Model Names ===
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
KIMI_MODEL_NAME = os.getenv("KIMI_MODEL_NAME", "moonshotai/kimi-k2-instruct")

# === Embeddings ===
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME", "models/gemini-embedding-001")

# === Whisper (Speech-to-Text) ===
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "whisper-large-v3-turbo")

# === Qdrant ===
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "recare_knowledge_base_v2")

# === RAG Hyperparameters ===
VECTOR_SEARCH_K = int(os.getenv("VECTOR_SEARCH_K", "10"))
FINAL_DOCS_K = int(os.getenv("FINAL_DOCS_K", "4"))

# === Telegram ===
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# === GCP Firestore ===
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "chat_sessions")

# === LangSmith ===
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-agent-v2")

# === Application Settings ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
