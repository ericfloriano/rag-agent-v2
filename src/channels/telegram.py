"""
Telegram Bot integration for Agentic RAG v2.

Handles incoming text and voice messages via python-telegram-bot.
Uses polling mode for local development.

Flow:
    User sends text → validate input → get history → run_agent → save messages → reply
    User sends voice → download OGG → transcribe → validate → get history → run_agent → save → reply
"""

import logging
import tempfile
from pathlib import Path

from telegram import Update
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from src.agent.graph import run_agent
from src.config import TELEGRAM_BOT_TOKEN, USE_WEBHOOK
from src.guardrails.input_guard import validate_input, InputGuardError
from src.guardrails.output_guard import sanitize_output
from src.memory.firestore import save_message, get_history, format_history
from src.memory.semantic_cache import check_cache, add_to_cache
from src.voice.transcriber import transcribe_audio

logger = logging.getLogger(__name__)

# Telegram bot application (module-level singleton)
_app: Application | None = None

def get_telegram_app() -> Application | None:
    """Retrieve the global Telegram application instance."""
    return _app


# ============================================================
# Command Handlers
# ============================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    welcome = (
        "👋 Olá! Sou o assistente técnico dos equipamentos **ReCARE** e **RePAD**.\n\n"
        "Posso te ajudar com:\n"
        "• Informações técnicas dos equipamentos\n"
        "• Procedimentos de operação e manutenção\n"
        "• Especificações e configurações\n"
        "• Solução de problemas\n\n"
        "Você pode me enviar mensagens de **texto** ou **áudio**! 🎤\n\n"
        "Como posso te ajudar hoje?"
    )
    await update.message.reply_text(welcome, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    help_text = (
        "📚 **Comandos disponíveis:**\n\n"
        "/start — Iniciar conversa\n"
        "/help — Ver esta ajuda\n\n"
        "💡 **Dicas:**\n"
        "• Envie perguntas sobre os equipamentos ReCARE e RePAD\n"
        "• Você pode enviar mensagens de áudio 🎤\n"
        "• Eu lembro das últimas mensagens da nossa conversa"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


# ============================================================
# Message Handlers
# ============================================================

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle incoming text messages.

    Flow: validate → get history → run agent → save messages → reply
    """
    chat_id = str(update.effective_chat.id)
    user_text = update.message.text

    logger.info(f"💬 Text message from chat {chat_id}: '{user_text[:80]}'")

    # 1. Validate input
    try:
        cleaned_text = validate_input(user_text)
    except InputGuardError as e:
        await update.message.reply_text(e.user_message)
        return

    # 2. Show typing indicator
    await update.effective_chat.send_action("typing")
    # 3. Check Semantic Cache
    cached_response = await check_cache(cleaned_text)
    if cached_response:
        response = cached_response
        logger.info(f"🌟 Cache Hit in Telegram! Bypassing LangGraph.")
    else:
        # 4. Get conversation history
        try:
            history = get_history(chat_id, limit=5)
            chat_history = format_history(history)
        except Exception as e:
            logger.warning(f"⚠️ Could not load history: {e}")
            chat_history = ""

        # 5. Run the agent
        try:
            response = await run_agent(
                question=cleaned_text,
                chat_history=chat_history,
            )
        except Exception as e:
            logger.error(f"❌ Agent error: {type(e).__name__}: {e}")
            response = (
                "Desculpe, ocorreu um erro ao processar sua pergunta. "
                "Por favor, tente novamente em alguns instantes. 🔄"
            )

        # 6. Sanitize output
        response = sanitize_output(response)
        
        # 7. Add to Cache
        if "Desculpe, ocorreu um erro" not in response:
            await add_to_cache(cleaned_text, response)

    # 6. Save messages to Firestore
    try:
        save_message(chat_id, "user", cleaned_text)
        save_message(chat_id, "assistant", response)
    except Exception as e:
        logger.warning(f"⚠️ Could not save to Firestore: {e}")

    # 7. Reply
    try:
        await update.message.reply_text(response, parse_mode="Markdown")
    except BadRequest as e:
        logger.error(f"⚠️ Telegram Markdown Error: {e}. Retrying without formatting.")
        await update.message.reply_text(response)

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle incoming voice messages.

    Flow: download OGG → transcribe → validate → get history → run agent → save → reply
    """
    chat_id = str(update.effective_chat.id)

    logger.info(f"🎤 Voice message from chat {chat_id}")

    # 1. Download voice file
    try:
        voice = update.message.voice
        voice_file = await context.bot.get_file(voice.file_id)

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            await voice_file.download_to_drive(str(tmp_path))

        logger.info(
            f"📥 Downloaded voice: {tmp_path.name} "
            f"({tmp_path.stat().st_size / 1024:.1f} KB)"
        )
    except Exception as e:
        logger.error(f"❌ Failed to download voice: {e}")
        await update.message.reply_text(
            "Desculpe, não consegui processar seu áudio. "
            "Tente enviar novamente. 🔄"
        )
        return

    # 2. Transcribe audio
    try:
        await update.effective_chat.send_action("typing")
        transcribed_text = await transcribe_audio(tmp_path)
        logger.info(f"📝 Transcription: '{transcribed_text[:80]}'")

        # Send transcription feedback to user
        await update.message.reply_text(
            f"🎤 _Entendi:_ \"{transcribed_text}\"",
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.error(f"❌ Transcription failed: {e}")
        await update.message.reply_text(
            "Desculpe, não consegui transcrever seu áudio. "
            "Tente enviar novamente ou digite sua pergunta. 🔄"
        )
        return
    finally:
        # Clean up temp file
        try:
            tmp_path.unlink()
        except Exception:
            pass

    # 3. Validate transcribed input
    try:
        cleaned_text = validate_input(transcribed_text)
    except InputGuardError as e:
        await update.message.reply_text(e.user_message)
        return

    # 4. Show typing indicator
    await update.effective_chat.send_action("typing")

    # 5. Check Semantic Cache
    cached_response = await check_cache(cleaned_text)
    if cached_response:
        response = cached_response
        logger.info(f"🌟 Cache Hit in Telegram Voice! Bypassing LangGraph.")
    else:
        # 6. Get conversation history
        try:
            history = get_history(chat_id, limit=5)
            chat_history = format_history(history)
        except Exception as e:
            logger.warning(f"⚠️ Could not load history: {e}")
            chat_history = ""

        # 7. Run the agent
        try:
            response = await run_agent(
                question=cleaned_text,
                chat_history=chat_history,
            )
        except Exception as e:
            logger.error(f"❌ Agent error: {type(e).__name__}: {e}")
            response = (
                "Desculpe, ocorreu um erro ao processar sua pergunta. "
                "Por favor, tente novamente em alguns instantes. 🔄"
            )

        # 8. Sanitize output
        response = sanitize_output(response)

        # 9. Add to Cache
        if "Desculpe, ocorreu um erro" not in response:
            await add_to_cache(cleaned_text, response)

    # 8. Save messages
    try:
        save_message(chat_id, "user", f"[áudio] {cleaned_text}")
        save_message(chat_id, "assistant", response)
    except Exception as e:
        logger.warning(f"⚠️ Could not save to Firestore: {e}")

    # 9. Reply
    try:
        await update.message.reply_text(response, parse_mode="Markdown")
    except BadRequest as e:
        logger.error(f"⚠️ Telegram Markdown Error: {e}. Retrying without formatting.")
        await update.message.reply_text(response)

# ============================================================
# Bot Lifecycle
# ============================================================

def create_telegram_app() -> Application:
    """
    Create and configure the Telegram bot application.

    Returns:
        Configured Application instance.
    """
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set. Cannot start Telegram bot.")

    logger.info("🤖 Creating Telegram bot application...")

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    # Register handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))

    logger.info("✅ Telegram bot configured with handlers: /start, /help, text, voice")
    return app


async def start_bot() -> Application:
    """
    Initialize and start the Telegram bot.
    Uses Webhook if configured, else falls back to Polling.

    Returns:
        Running Application instance.
    """
    global _app
    _app = create_telegram_app()

    # Initialize the application
    await _app.initialize()
    await _app.start()

    if USE_WEBHOOK:
        logger.info("🚀 Telegram bot initialized in WEBHOOK mode. Waiting for requests.")
        # Em modo webhook, a aplicação não bloqueia. O FastAPI injetará as requisições.
    else:
        # Start polling (non-blocking)
        await _app.updater.start_polling(drop_pending_updates=True)
        logger.info("🚀 Telegram bot started in POLLING mode!")

    return _app


async def stop_bot():
    """Stop the Telegram bot gracefully."""
    global _app
    if _app:
        logger.info("🛑 Stopping Telegram bot...")
        await _app.updater.stop()
        await _app.stop()
        await _app.shutdown()
        _app = None
        logger.info("✅ Telegram bot stopped.")
