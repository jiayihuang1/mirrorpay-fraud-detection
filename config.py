"""
Centralized configuration — single source of truth for all settings.
Import from here, no magic strings anywhere else.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Model IDs — swap here for final eval runs, no code changes needed
# Development defaults (cheap, fast, good enough for iteration)
# ---------------------------------------------------------------------------
MODEL_TRANSCRIPTION = os.getenv("MODEL_TRANSCRIPTION", "google/gemini-2.5-flash")
MODEL_TXN_AGENT = os.getenv("MODEL_TXN_AGENT", "google/gemini-2.5-flash")
MODEL_COMMS_AGENT = os.getenv("MODEL_COMMS_AGENT", "google/gemini-2.5-flash")
MODEL_ORCHESTRATOR = os.getenv("MODEL_ORCHESTRATOR", "deepseek/deepseek-r1-0528")
MODEL_MEMORY_AGENT = os.getenv("MODEL_MEMORY_AGENT", "google/gemini-2.5-flash")

# Shared model params — temperature low for consistency
MODEL_PARAMS = {
    "max_tokens": int(os.getenv("MAX_TOKENS", "8192")),
    "temperature": float(os.getenv("TEMPERATURE", "0.1")),
}

# ---------------------------------------------------------------------------
# Batch sizes — halved automatically on context overflow (400)
# ---------------------------------------------------------------------------
BATCH_SIZE_TXN_AGENT = int(os.getenv("BATCH_SIZE_TXN_AGENT", "50"))
BATCH_SIZE_ORCHESTRATOR = int(os.getenv("BATCH_SIZE_ORCHESTRATOR", "20"))
BATCH_SIZE_COMMS_AGENT = int(os.getenv("BATCH_SIZE_COMMS_AGENT", "20"))
COMMS_WINDOW_DAYS_BEFORE = int(os.getenv("COMMS_WINDOW_DAYS_BEFORE", "45"))
COMMS_WINDOW_DAYS_AFTER = int(os.getenv("COMMS_WINDOW_DAYS_AFTER", "3"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# ---------------------------------------------------------------------------
# Langfuse Tracing
# ---------------------------------------------------------------------------
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "data")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
MEMORY_DIR = os.getenv("MEMORY_DIR", "memory")  # threat intelligence briefs per level

# ---------------------------------------------------------------------------
# Dataset level → folder mapping
# Add levels 4-5 here when they are unlocked
# ---------------------------------------------------------------------------
LEVEL_FOLDERS: dict[int, str] = {
    1: "The Truman Show - train",
    2: "Brave New World - train",
    3: "Deus Ex - train",
}

# ---------------------------------------------------------------------------
# Dataset schema — column names from actual data inspection
# ---------------------------------------------------------------------------
COL_TXN_ID = "transaction_id"
COL_SENDER_ID = "sender_id"
COL_RECIPIENT_ID = "recipient_id"
COL_TXN_TYPE = "transaction_type"
COL_AMOUNT = "amount"
COL_LOCATION = "location"
COL_PAYMENT_METHOD = "payment_method"
COL_SENDER_IBAN = "sender_iban"
COL_RECIPIENT_IBAN = "recipient_iban"
COL_BALANCE_AFTER = "balance_after"
COL_DESCRIPTION = "description"
COL_TIMESTAMP = "timestamp"
