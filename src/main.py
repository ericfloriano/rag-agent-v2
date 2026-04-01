"""
Main FastAPI application entrypoint.

Orchestrates the HTTP server, the Admin Panel router, and the Telegram Bot.
Uses modern FastAPI lifespan to manage the startup and shutdown.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from telegram import Update

# --- Security Imports ---
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from src.admin.security import limiter

from src.config import USE_WEBHOOK, WEBHOOK_URL
from src.channels.telegram import start_bot, stop_bot, get_telegram_app
from src.admin.router import router as admin_router

# Configure base logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up Agentic RAG v2...")
    
    try:
        app_bot = await start_bot()
        if USE_WEBHOOK and WEBHOOK_URL:
            webhook_endpoint = f"{WEBHOOK_URL}/webhook"
            await app_bot.bot.set_webhook(url=webhook_endpoint)
            logger.info(f"🔗 Webhook set to: {webhook_endpoint}")
            
    except Exception as e:
        logger.error(f"❌ Failed to start Telegram bot: {e}")

    yield  

    logger.info("🛑 Shutting down Agentic RAG v2...")
    try:
        await stop_bot()
    except Exception as e:
        logger.error(f"❌ Error during Telegram bot shutdown: {e}")


app = FastAPI(
    title="Agentic RAG v2 API",
    description="Backend for the Visuri/Contourline AI Agent and Admin Panel.",
    version="2.0.0",
    lifespan=lifespan,
)

# --- Restore SlowAPI Setup ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin_router)


@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "agent": "Agentic RAG v2 is running."}


@app.post("/webhook", tags=["Telegram Integration"])
async def telegram_webhook(request: Request):
    if not USE_WEBHOOK:
        return Response(status_code=status.HTTP_403_FORBIDDEN)

    telegram_app = get_telegram_app()
    if not telegram_app:
        logger.error("⚠️ Webhook received but Telegram app is not initialized.")
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    try:
        payload = await request.json()
        update = Update.de_json(payload, telegram_app.bot)
        await telegram_app.process_update(update)
        return Response(status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"❌ Error processing webhook update: {e}")
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
