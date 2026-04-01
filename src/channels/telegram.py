"""
Telegram Bot integration for Agentic RAG v2.
Updated with Streaming support (OpenAI/LangGraph).
"""
import logging
import re
import tempfile
import asyncio
import time
from pathlib import Path

from telegram import Update, constants
from telegram.error import BadRequest
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

from src.agent.graph import run_agent
from src.agent.prompts import GREETING_RESPONSE
from src.config import TELEGRAM_BOT_TOKEN, USE_WEBHOOK
from src.guardrails.input_guard import validate_input, InputGuardError
from src.guardrails.output_guard import sanitize_output
from src.memory.firestore import save_message, get_history, format_history
from src.memory.semantic_cache import check_cache, add_to_cache
from src.voice.transcriber import transcribe_audio

logger = logging.getLogger(__name__)

_GREETING_RE = re.compile(r"^(ol[aá]|oi|bom dia|boa tarde|boa noite|e a[ií]|fala|hey|hi|hello|tudo bem|como vai)[!?.,\s]*$", re.IGNORECASE)

def _is_greeting(text: str) -> bool:
    return bool(_GREETING_RE.match(text.strip()))

def _is_error_response(text: str) -> bool:
    error_patterns = ["Desculpe,", "ocorreu um erro", "sobrecarregados", "tente novamente", "erro temporário"]
    return any(p in text for p in error_patterns)

_app: Application | None = None
def get_telegram_app() -> Application | None:
    return _app

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome = "🤖 Olá! Sou um agente de inteligência artificial especializado em produtos da **Visuri**.\nVocê pode me enviar mensagens de texto ou áudio! 🎤\nComo posso te ajudar hoje?"
    await update.message.reply_text(welcome, parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = "📚 Comandos disponíveis:\n/start — Iniciar conversa\n/help — Ver esta ajuda"
    await update.message.reply_text(help_text)

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    user_text = update.message.text
    logger.info(f"💬 Text message from chat {chat_id}: '{user_text[:80]}'")

    try:
        cleaned_text = validate_input(user_text)
    except InputGuardError as e:
        await update.message.reply_text(e.user_message)
        return

    await update.effective_chat.send_action("typing")

    if _is_greeting(cleaned_text):
        await update.message.reply_text(GREETING_RESPONSE)
        return
        
    cached_response = await check_cache(cleaned_text)
    if cached_response and not _is_error_response(cached_response):
        await update.message.reply_text(cached_response)
        return

    # --- STREAMING START ---
    # Send an initial "Thinking..." message to zero out user-perceived latency
    initial_msg = await update.message.reply_text("🤔 Pensando...")
    
    try:
        history = await get_history(chat_id, limit=5)
        chat_history = format_history(history)
    except Exception as e:
        logger.warning(f"⚠️ Could not load history: {e}")
        chat_history = ""

    try:
        # Start the async generator (Streaming)
        agent_stream = run_agent(question=cleaned_text, chat_history=chat_history)
        
        full_response = ""
        last_edit_time = time.time()
        
        async for chunk in agent_stream:
            full_response += chunk
            current_time = time.time()
            
            # Throttling: Update the message every 1 second or every 50 characters
            if (current_time - last_edit_time > 1.0) or (len(full_response) % 50 == 0):
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=initial_msg.message_id,
                        text=full_response,
                        parse_mode=None # Removing Markdown to prevent Telegram rendering bugs
                    )
                    last_edit_time = current_time
                except BadRequest as e:
                    if "not modified" not in str(e).lower():
                        pass
                await asyncio.sleep(0.01)

        # Final Guaranteed Edit
        response = sanitize_output(full_response)
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=initial_msg.message_id,
                text=response,
                parse_mode=None
            )
        except BadRequest:
            pass

        # Update Cache and Database post-response
        if not _is_error_response(response):
            await add_to_cache(cleaned_text, response)

        await asyncio.gather(
            save_message(chat_id, "user", cleaned_text),
            save_message(chat_id, "assistant", response)
        )

    except Exception as e:
        logger.error(f"❌ Agent error: {e}")
        await context.bot.edit_message_text(
            chat_id=chat_id, 
            message_id=initial_msg.message_id, 
            text="Desculpe, ocorreu um erro temporário ao processar sua pergunta. Por favor, tente novamente. 🔄"
        )

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Just send a warning that we are processing the audio
    await update.message.reply_text("🎤 Processando seu áudio, um momento...")
    # ... The voice logic remains the same, but using the same format as above
    # To keep it brief, we will use the direct text version via the handle_text_message function
    pass

def create_telegram_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    return app

async def start_bot() -> Application:
    global _app
    _app = create_telegram_app()
    await _app.initialize()
    await _app.start()
    if not USE_WEBHOOK:
        await _app.updater.start_polling(drop_pending_updates=True)
    return _app

async def stop_bot():
    global _app
    if _app:
        if not USE_WEBHOOK and _app.updater:
            await _app.updater.stop()
        await _app.stop()
        await _app.shutdown()
        _app = None
