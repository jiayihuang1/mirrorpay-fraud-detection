"""
Audio transcription — converts MP3 files to text using Gemini Flash Lite.
Transcripts are cached to disk and never re-transcribed on retry.
"""

import base64
import logging
import re
from pathlib import Path

from openai import OpenAI

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL_TRANSCRIPTION

logger = logging.getLogger(__name__)

# Filename pattern: YYYYMMDD_HHMMSS-<speaker_name>.mp3
_AUDIO_FILENAME_RE = re.compile(r"^\d{8}_\d{6}-(.+)\.mp3$", re.IGNORECASE)


def _speaker_name(filename: str) -> str:
    """Extract speaker name from audio filename convention.

    Args:
        filename: e.g. '20870117_010505-guido_döhn.mp3'

    Returns:
        Speaker name, e.g. 'guido_döhn', or the stem if pattern doesn't match.
    """
    m = _AUDIO_FILENAME_RE.match(Path(filename).name)
    return m.group(1) if m else Path(filename).stem


def transcribe_audio_files(audio_dir: Path) -> dict[str, str]:
    """Transcribe all MP3 files in a directory, caching results to disk.

    Args:
        audio_dir: Directory containing .mp3 files.

    Returns:
        Dict mapping speaker_name to transcript text.
        Multiple files for the same speaker are concatenated.
    """
    if not audio_dir.exists():
        return {}

    mp3_files = sorted(audio_dir.glob("*.mp3"))
    if not mp3_files:
        return {}

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    transcripts: dict[str, list[str]] = {}

    for mp3_path in mp3_files:
        cache_path = mp3_path.with_suffix(".txt")
        speaker = _speaker_name(mp3_path.name)

        if cache_path.exists():
            logger.info("transcription cache hit: %s", mp3_path.name)
            text = cache_path.read_text(encoding="utf-8")
        else:
            logger.info("transcribing: %s", mp3_path.name)
            text = _transcribe_one(client, mp3_path)
            cache_path.write_text(text, encoding="utf-8")
            logger.info("cached transcript: %s", cache_path)

        transcripts.setdefault(speaker, []).append(
            f"[Recording: {mp3_path.name}]\n{text}"
        )

    return {speaker: "\n\n".join(parts) for speaker, parts in transcripts.items()}


def _transcribe_one(client: OpenAI, mp3_path: Path) -> str:
    """Send one MP3 to Gemini Flash Lite and return the transcript.

    Args:
        client: Configured OpenAI client pointing at OpenRouter.
        mp3_path: Path to the MP3 file.

    Returns:
        Raw transcript text.
    """
    audio_b64 = base64.b64encode(mp3_path.read_bytes()).decode()

    response = client.chat.completions.create(
        model=MODEL_TRANSCRIPTION,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "mp3"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Transcribe this audio recording exactly as spoken. "
                            "Output only the raw transcript — no labels, "
                            "no timestamps, no commentary."
                        ),
                    },
                ],
            }
        ],
        max_tokens=2000,
    )
    return response.choices[0].message.content or ""
