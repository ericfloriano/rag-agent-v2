"""
Whisper Speech-to-Text transcription via Groq API.
Converts audio files (OGG from Telegram) to text using Whisper Large v3 Turbo.
"""

import logging
from pathlib import Path

from groq import Groq

from src.config import GROQ_API_KEY, WHISPER_MODEL_NAME

logger = logging.getLogger(__name__)


def get_groq_client() -> Groq:
    """Initialize the Groq client for Whisper."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found. Cannot use Whisper transcription.")
    return Groq(api_key=GROQ_API_KEY)


async def transcribe_audio(audio_path: str | Path) -> str:
    """
    Transcribe an audio file using Groq's Whisper API.

    Args:
        audio_path: Path to the audio file (OGG, MP3, WAV, etc.)

    Returns:
        Transcribed text string.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"🎤 Transcribing audio: {audio_path.name} ({audio_path.stat().st_size / 1024:.1f} KB)")

    client = get_groq_client()

    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(audio_path.name, audio_file.read()),
            model=WHISPER_MODEL_NAME,
            language="pt",  # Portuguese (Brazil)
            response_format="text",
        )

    text = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
    logger.info(f"✅ Transcription complete: '{text[:80]}...'")
    return text
