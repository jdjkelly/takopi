"""Audio transcription using whisper.cpp via pywhispercpp."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import anyio.to_thread

from .logging import get_logger

if TYPE_CHECKING:
    from pywhispercpp.model import Model

logger = get_logger(__name__)

_model: Model | None = None
MODEL_NAME = "tiny.en"


def _get_model() -> Model:
    """Lazy-load the whisper model."""
    global _model
    if _model is None:
        from pywhispercpp.model import Model

        logger.info("transcribe.loading_model", model=MODEL_NAME)
        _model = Model(MODEL_NAME)
        logger.info("transcribe.model_loaded", model=MODEL_NAME)
    return _model


async def transcribe_audio(audio_bytes: bytes, *, suffix: str = ".ogg") -> str | None:
    """Transcribe audio bytes to text.

    Args:
        audio_bytes: Raw audio data (OGG/OPUS from Telegram voice messages)
        suffix: File extension hint for the audio format

    Returns:
        Transcribed text, or None if transcription failed
    """

    def _transcribe() -> str | None:
        try:
            model = _get_model()

            # Write to temp file (whisper.cpp needs a file path)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_bytes)
                temp_path = Path(f.name)

            try:
                logger.debug("transcribe.start", path=str(temp_path), size=len(audio_bytes))
                segments = model.transcribe(str(temp_path))
                text = " ".join(seg.text.strip() for seg in segments).strip()
                logger.info("transcribe.done", text_len=len(text))
                return text if text else None
            finally:
                temp_path.unlink(missing_ok=True)

        except Exception as e:
            logger.error(
                "transcribe.error",
                error=str(e),
                error_type=e.__class__.__name__,
            )
            return None

    # Run in thread pool to avoid blocking the event loop
    return await anyio.to_thread.run_sync(_transcribe)
