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

    # 1. Initialize LangGraph Agent (lazy — compiled on first use)
    from src.agent.graph import get_graph
    logger.info("🧠 LangGraph agent ready (lazy initialization)")

    # 2. Initialize Firestore
    try:
        from src.memory.firestore import get_firestore_client
        get_firestore_client()
        logger.info("🔥 Firestore connected")
    except Exception as e:
        logger.warning(f"⚠️ Firestore not available: {e}")

    # 3. Start Telegram Bot
    from src.channels.telegram import start_bot, stop_bot
    try:
        await start_bot()
        logger.info("📱 Telegram bot started")
    except Exception as e:
        logger.error(f"❌ Failed to start Telegram bot: {e}")

    logger.info("✅ Agentic RAG v2 started successfully!")

    yield  # Application runs here

    # --- SHUTDOWN ---
    logger.info("🛑 Shutting down Agentic RAG v2...")
    try:
        await stop_bot()
    except Exception as e:
        logger.warning(f"⚠️ Error stopping Telegram bot: {e}")
    logger.info("👋 Agentic RAG v2 shut down.")


app = FastAPI(
    title="Agentic RAG v2",
    description="Next-gen Retrieval-Augmented Generation with LangGraph, Multi-LLM Cascade, and Voice Support",
    version="2.0.0",
    lifespan=lifespan,
)

# Setup SlowAPI Rate Limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from src.admin.security import limiter

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include Admin Router
from src.admin.router import router as admin_router
app.include_router(admin_router)


@app.get("/")
async def health_check():
    """Health check endpoint for Cloud Run."""
    return {"status": "healthy", "service": "rag-agent-v2"}


@app.get("/readiness")
async def readiness():
    """Readiness probe for Cloud Run. Checks if all services are connected."""
    # TODO: Add actual checks (Qdrant, LLM, Firestore)
    return {"status": "ready"}


@app.post("/webhook")
async def telegram_webhook(request: Request):
    """
    Handle incoming Telegram updates (called by Telegram servers).
    Requires USE_WEBHOOK=True in production.
    """
    from telegram import Update
    from src.channels.telegram import get_telegram_app

    app_telegram = get_telegram_app()
    if not app_telegram:
        logger.error("Webhook called but Telegram app is not initialized.")
        return Response(status_code=500, content="Bot not ready")

    try:
        data = await request.json()
        update = Update.de_json(data, app_telegram.bot)
        await app_telegram.process_update(update)
    except Exception as e:
        logger.error(f"Failed to process webhook update: {e}")
        return Response(status_code=400, content="Bad Request")

    return Response(status_code=200, content="OK")
