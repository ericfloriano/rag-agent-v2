"""
FastAPI entrypoint for the Agentic RAG v2 application.
Serves as the webhook receiver for Telegram and health check endpoint for Cloud Run.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from src.config import LOG_LEVEL, TELEGRAM_BOT_TOKEN

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    logger.info("🚀 Starting Agentic RAG v2...")

    # --- STARTUP ---
    # These will be initialized in later phases:
    # 1. Load LangGraph Agent
    # 2. Initialize Telegram Bot
    # 3. Connect to Qdrant Cloud
    # 4. Connect to Firestore
    logger.info("✅ Agentic RAG v2 started successfully!")

    yield  # Application runs here

    # --- SHUTDOWN ---
    logger.info("🛑 Shutting down Agentic RAG v2...")


app = FastAPI(
    title="Agentic RAG v2",
    description="Next-gen Retrieval-Augmented Generation with LangGraph, Multi-LLM Cascade, and Voice Support",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def health_check():
    """Health check endpoint for Cloud Run."""
    return {"status": "healthy", "service": "rag-agent-v2"}


@app.get("/readiness")
async def readiness():
    """Readiness probe for Cloud Run. Checks if all services are connected."""
    # TODO: Add actual checks (Qdrant, LLM, Firestore)
    return {"status": "ready"}
