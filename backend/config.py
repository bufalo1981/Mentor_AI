"""Settings for Mentor AI, read once from environment variables (see .env.example)."""

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(ROOT_DIR / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Models — Gemini 2.5 text models return 404 for new API keys; default to Gemini 3.x Flash. Override any in .env.
TEXT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
# Pinned on purpose: the "-latest" alias of the 2.5 native-audio model closes the session (1007 "audio content
# type is not supported") as soon as the candidate speaks when function tools are declared. Tested Oct 2026.
LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
TRANSCRIBE_MODEL = os.getenv("GEMINI_TRANSCRIBE_MODEL", TEXT_MODEL)
INTERVIEWER_VOICE = os.getenv("INTERVIEWER_VOICE", "Charon")


def _question_count() -> int:
    """Number of mock interview questions: default 7, clamped to 5–10."""
    try:
        value = int(os.getenv("INTERVIEW_QUESTION_COUNT", "7"))
    except ValueError:
        value = 7
    return max(5, min(10, value))


QUESTION_COUNT = _question_count()

# "live" = Gemini Live API (real-time voice). "turn" = turn-by-turn fallback (TTS + recorded answers).
VOICE_MODE = os.getenv("VOICE_MODE", "live") if os.getenv("VOICE_MODE", "live") in ("live", "turn") else "live"

# Job search looks back this many days ("past 3 months").
JOB_LOOKBACK_DAYS = 90
JOBS_PER_ROUND = 5

DATA_DIR = Path(os.getenv("MENTOR_DATA_DIR", ROOT_DIR / "data"))
if not DATA_DIR.is_absolute():
    DATA_DIR = ROOT_DIR / DATA_DIR
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "mentor.db"
SAMPLE_RESUME = ROOT_DIR / "samples" / "sample_resume.pdf"

MAX_RESUME_BYTES = 10 * 1024 * 1024
RESUME_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".txt": "text/plain",
}
