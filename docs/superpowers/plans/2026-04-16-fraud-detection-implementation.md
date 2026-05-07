# Fraud Detection System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete multi-agent fraud detection pipeline that processes transaction datasets and outputs a list of fraudulent transaction UUIDs.

**Architecture:** Zero-cost statistical pre-processing feeds a dynamic strands orchestrator agent that selectively calls Transaction, Network, Location, and Communications tools before making each binary fraud classification. A Memory Agent synthesises patterns after each level to inform the next.

**Tech Stack:** Python 3.11+, `strands-agents`, `openai` client, `pandas`, `langfuse`, OpenRouter API, `pytest`

---

## File Map

| File | Responsibility |
|------|---------------|
| `config.py` | All constants — already updated |
| `src/utils.py` | `make_model`, `call_with_retry`, `validate_output`, `estimate_tokens`, `UUID_PATTERN` |
| `src/analysis.py` | Statistical signal computation: baselines, anomalies, graph, text formatter |
| `src/transcription.py` | Audio → text via Gemini Flash Lite, disk cache |
| `src/sub_agents.py` | `run_comms_agent`, `run_transaction_agent` (callable as strands tools) |
| `src/network.py` | `get_network_context` (dataset-level IBAN analysis for a given IBAN) |
| `src/orchestrator.py` | Dynamic strands Agent with tools, `parse_classifications` |
| `src/memory.py` | `run_memory_agent`, `load_threat_intel` |
| `main.py` | `run_level`, `main` CLI |
| `tests/test_utils.py` | Unit tests for `validate_output`, `estimate_tokens` |
| `tests/test_analysis.py` | Unit tests for signal computation, formatter |
| `tests/test_orchestrator.py` | Unit tests for `parse_classifications` |
| `tests/test_memory.py` | Unit tests for `load_threat_intel` |

---

## Task 1: Complete `src/utils.py`

**Files:**
- Modify: `src/utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_utils.py
import pytest
from src.utils import validate_output, estimate_tokens, UUID_PATTERN

TXN_A = "e1021ab7-c2de-4791-994b-bab86e6fbe3e"
TXN_B = "8830a720-ff34-4dce-a578-e5b8006b2976"
TXN_C = "1c6db202-22d8-443f-86e7-fb1a8df05e84"


def test_validate_output_finds_present_ids():
    text = f"{TXN_A}: 1 | high | suspicious\n{TXN_B}: 0 | low | normal"
    found, missing = validate_output(text, {TXN_A, TXN_B})
    assert found == {TXN_A, TXN_B}
    assert missing == set()


def test_validate_output_detects_missing_ids():
    text = f"{TXN_A}: 1 | high | suspicious"
    found, missing = validate_output(text, {TXN_A, TXN_B})
    assert found == {TXN_A}
    assert missing == {TXN_B}


def test_validate_output_case_insensitive():
    text = f"{TXN_A.upper()}: 1 | high | reason"
    found, missing = validate_output(text, {TXN_A})
    assert found == {TXN_A}
    assert missing == set()


def test_estimate_tokens_rough_heuristic():
    text = "a" * 400
    assert estimate_tokens(text) == 100


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/weifong.chia/Documents/MSc_AI/hackathon/reply-ai
source .venv/bin/activate
python -m pytest tests/test_utils.py -v
```

Expected: `ImportError` or `AttributeError` — functions not fully implemented yet.

- [ ] **Step 3: Rewrite `src/utils.py` with full implementation**

```python
"""
Utility functions: model factory, retry wrapper, output validation.
"""

import re
import time
import random
import logging
from typing import Callable

from langfuse import Langfuse
from strands.models.openai import OpenAIModel

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODEL_PARAMS,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST,
    MAX_RETRIES,
)

logger = logging.getLogger(__name__)

UUID_PATTERN = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)


class ContextOverflowError(Exception):
    """Raised on HTTP 400 — signals caller to halve batch and retry."""


def make_model(model_id: str, max_tokens: int | None = None) -> OpenAIModel:
    """Create an OpenRouter-routed OpenAI-compatible model instance.

    Args:
        model_id: Model identifier, e.g. 'qwen/qwen3-30b-a3b'.
        max_tokens: Override default output token limit.

    Returns:
        Configured OpenAIModel instance.
    """
    return OpenAIModel(
        client_args={
            "api_key": OPENROUTER_API_KEY,
            "base_url": OPENROUTER_BASE_URL,
        },
        model_id=model_id,
        params={
            "max_tokens": max_tokens or MODEL_PARAMS["max_tokens"],
            "temperature": MODEL_PARAMS["temperature"],
        },
    )


def estimate_tokens(text: str) -> int:
    """Rough token count estimate: 4 chars ≈ 1 token.

    Args:
        text: Input string.

    Returns:
        Estimated token count.
    """
    return len(text) // 4


def validate_output(
    response_text: str,
    expected_ids: set[str],
) -> tuple[set[str], set[str]]:
    """Check which expected transaction UUIDs appear in the response.

    Args:
        response_text: Raw LLM output text.
        expected_ids: Set of transaction UUIDs that should appear.

    Returns:
        Tuple of (found_ids, missing_ids) — both lowercased sets.
    """
    found_ids = {m.lower() for m in UUID_PATTERN.findall(response_text)}
    expected_lower = {e.lower() for e in expected_ids}
    found = found_ids & expected_lower
    missing = expected_lower - found
    return found, missing


def call_with_retry(
    call_fn: Callable[[], str],
    label: str = "llm_call",
    max_retries: int = MAX_RETRIES,
) -> str:
    """Call call_fn() with retry logic for all API failure modes.

    Raises ContextOverflowError on HTTP 400 so the caller can halve
    the batch and retry — never falls back to non-LLM logic.

    Args:
        call_fn: Zero-argument callable that returns the LLM response string.
        label: Identifier for log messages (e.g. 'comms_agent').
        max_retries: Maximum retry attempts for transient errors.

    Returns:
        Raw response string from the LLM.

    Raises:
        ContextOverflowError: On HTTP 400 (context window exceeded).
        Exception: After max_retries exhausted for other errors.
    """
    import openai

    for attempt in range(max_retries + 1):
        try:
            return call_fn()
        except openai.BadRequestError as e:
            logger.warning("%s: context overflow (400) on attempt %d: %s",
                           label, attempt + 1, e)
            raise ContextOverflowError(str(e)) from e
        except openai.RateLimitError as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            logger.warning("%s: rate limited (429), waiting %.1fs (attempt %d/%d)",
                           label, wait, attempt + 1, max_retries)
            if attempt < max_retries:
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            logger.warning("%s: %s on attempt %d/%d, retrying in %.1fs",
                           label, type(e).__name__, attempt + 1, max_retries, wait)
            if attempt < max_retries:
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"{label}: exhausted {max_retries} retries")  # unreachable


def make_langfuse_client() -> Langfuse:
    """Create an initialized Langfuse client for tracing.

    Returns:
        Configured Langfuse client.

    Raises:
        ValueError: If Langfuse credentials are missing from environment.
    """
    if not all([LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY]):
        raise ValueError("Missing Langfuse credentials in .env")
    return Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_utils.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils.py tests/test_utils.py
git commit -m "feat: implement utils — call_with_retry, validate_output, estimate_tokens"
```

---

## Task 2: Statistical signals — baselines and graph (`src/analysis.py`)

**Files:**
- Modify: `src/analysis.py`
- Create: `tests/test_analysis.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_analysis.py
import pandas as pd
import pytest
from src.analysis import (
    load_transactions,
    compute_user_baselines,
    compute_graph_signals,
)

@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "transaction_id": "e1021ab7-c2de-4791-994b-bab86e6fbe3e",
            "sender_id": "USER_A",
            "recipient_id": "USER_B",
            "transaction_type": "transfer",
            "amount": "100.0",
            "location": "",
            "payment_method": "",
            "sender_iban": "IT1234",
            "recipient_iban": "DE5678",
            "balance_after": "900.0",
            "description": "Rent",
            "timestamp": "2087-01-01T10:00:00",
        },
        {
            "transaction_id": "8830a720-ff34-4dce-a578-e5b8006b2976",
            "sender_id": "USER_A",
            "recipient_id": "USER_B",
            "transaction_type": "transfer",
            "amount": "200.0",
            "location": "",
            "payment_method": "",
            "sender_iban": "IT1234",
            "recipient_iban": "DE5678",
            "balance_after": "700.0",
            "description": "Rent 2",
            "timestamp": "2087-01-02T11:00:00",
        },
        {
            "transaction_id": "1c6db202-22d8-443f-86e7-fb1a8df05e84",
            "sender_id": "USER_C",
            "recipient_id": "USER_B",
            "transaction_type": "transfer",
            "amount": "5000.0",
            "location": "",
            "payment_method": "",
            "sender_iban": "FR9999",
            "recipient_iban": "DE5678",
            "balance_after": "200.0",
            "description": "Suspicious",
            "timestamp": "2087-01-02T03:00:00",
        },
    ])


def test_compute_user_baselines_mean(sample_df):
    from src.analysis import _coerce_types
    df = _coerce_types(sample_df)
    baselines = compute_user_baselines(df)
    assert "USER_A" in baselines
    assert baselines["USER_A"]["mean_amount"] == pytest.approx(150.0)


def test_compute_user_baselines_known_recipients(sample_df):
    from src.analysis import _coerce_types
    df = _coerce_types(sample_df)
    baselines = compute_user_baselines(df)
    assert "USER_B" in baselines["USER_A"]["known_recipients"]


def test_compute_graph_signals_shared_iban(sample_df):
    from src.analysis import _coerce_types
    df = _coerce_types(sample_df)
    graph = compute_graph_signals(df)
    # DE5678 is recipient of both USER_A and USER_C — shared_iban_count = 2
    txn_c = "1c6db202-22d8-443f-86e7-fb1a8df05e84"
    assert graph[txn_c]["shared_iban_count"] == 2


def test_compute_graph_signals_circular(sample_df):
    from src.analysis import _coerce_types
    df = _coerce_types(sample_df)
    graph = compute_graph_signals(df)
    # No circular transfers in sample data
    for txn_id, sig in graph.items():
        assert sig["circular_transfer"] is False
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_analysis.py -v
```

Expected: `ImportError` — functions not yet defined.

- [ ] **Step 3: Implement `src/analysis.py` — data loading, baselines, graph signals**

```python
"""
Statistical signal computation — zero LLM cost.
Computes per-user baselines and per-transaction anomaly signals from raw data.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    COL_TXN_ID, COL_SENDER_ID, COL_RECIPIENT_ID, COL_TXN_TYPE,
    COL_AMOUNT, COL_LOCATION, COL_PAYMENT_METHOD, COL_SENDER_IBAN,
    COL_RECIPIENT_IBAN, COL_BALANCE_AFTER, COL_DESCRIPTION, COL_TIMESTAMP,
)

logger = logging.getLogger(__name__)


def load_transactions(data_dir: Path) -> pd.DataFrame:
    """Load and type-coerce transactions.csv.

    Args:
        data_dir: Path to the dataset directory.

    Returns:
        DataFrame with numeric amount/balance and parsed timestamps.
    """
    df = pd.read_csv(data_dir / "transactions.csv")
    return _coerce_types(df)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parse numeric and datetime columns."""
    df = df.copy()
    df[COL_AMOUNT] = pd.to_numeric(df[COL_AMOUNT], errors="coerce")
    df[COL_BALANCE_AFTER] = pd.to_numeric(df[COL_BALANCE_AFTER], errors="coerce")
    df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP])
    return df


def compute_user_baselines(df: pd.DataFrame) -> dict[str, dict]:
    """Compute per-user baseline statistics across their transaction history.

    Args:
        df: Transactions DataFrame (type-coerced).

    Returns:
        Dict mapping sender_id to baseline metrics dict.
    """
    baselines: dict[str, dict] = {}
    for user_id, group in df.groupby(COL_SENDER_ID):
        amounts = group[COL_AMOUNT].dropna()
        duration_days = max(
            (group[COL_TIMESTAMP].max() - group[COL_TIMESTAMP].min()).days, 1
        )
        baselines[str(user_id)] = {
            "mean_amount": float(amounts.mean()) if len(amounts) > 0 else 0.0,
            "std_amount": float(amounts.std()) if len(amounts) > 1 else 1.0,
            "typical_hours": group[COL_TIMESTAMP].dt.hour.value_counts().to_dict(),
            "typical_types": group[COL_TXN_TYPE].value_counts(normalize=True).to_dict(),
            "known_recipients": set(group[COL_RECIPIENT_ID].dropna().tolist()),
            "known_recipient_ibans": set(group[COL_RECIPIENT_IBAN].dropna().tolist()),
            "avg_daily_velocity": len(group) / duration_days,
        }
    return baselines


def compute_graph_signals(df: pd.DataFrame) -> dict[str, dict]:
    """Compute dataset-level IBAN graph anomaly signals.

    Args:
        df: Transactions DataFrame (type-coerced).

    Returns:
        Dict mapping transaction_id to graph signal dict.
    """
    iban_sender_counts = (
        df.dropna(subset=[COL_RECIPIENT_IBAN, COL_SENDER_ID])
        .groupby(COL_RECIPIENT_IBAN)[COL_SENDER_ID]
        .nunique()
    )
    circular_ids = _detect_circular_transfers(df)

    graph: dict[str, dict] = {}
    for _, row in df.iterrows():
        txn_id = row[COL_TXN_ID]
        iban = row.get(COL_RECIPIENT_IBAN, "")
        count = int(iban_sender_counts.get(iban, 1)) if iban else 1
        graph[txn_id] = {
            "shared_iban_count": count,
            "circular_transfer": txn_id in circular_ids,
        }
    return graph


def _detect_circular_transfers(df: pd.DataFrame) -> set[str]:
    """Detect A→B→A transfer patterns within a 24-hour window.

    Args:
        df: Transactions DataFrame (type-coerced).

    Returns:
        Set of transaction_ids involved in circular patterns.
    """
    circular: set[str] = set()
    transfers = df[df[COL_TXN_TYPE] == "transfer"].sort_values(COL_TIMESTAMP)

    for _, row in transfers.iterrows():
        t_start = row[COL_TIMESTAMP]
        t_end = t_start + pd.Timedelta(hours=24)
        sender, recipient = row[COL_SENDER_ID], row[COL_RECIPIENT_ID]

        reverse = transfers[
            (transfers[COL_SENDER_ID] == recipient)
            & (transfers[COL_RECIPIENT_ID] == sender)
            & (transfers[COL_TIMESTAMP] > t_start)
            & (transfers[COL_TIMESTAMP] <= t_end)
        ]
        if not reverse.empty:
            circular.add(row[COL_TXN_ID])
            circular.update(reverse[COL_TXN_ID].tolist())
    return circular
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_analysis.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/analysis.py tests/test_analysis.py
git commit -m "feat: statistical signals — user baselines and IBAN graph anomalies"
```

---

## Task 3: Per-transaction signals and text formatter (`src/analysis.py`)

**Files:**
- Modify: `src/analysis.py` (add functions)
- Modify: `tests/test_analysis.py` (add tests)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_analysis.py`:

```python
def test_compute_statistical_signals_amount_zscore(sample_df):
    from src.analysis import _coerce_types, compute_user_baselines, compute_graph_signals, compute_statistical_signals
    df = _coerce_types(sample_df)
    baselines = compute_user_baselines(df)
    graph = compute_graph_signals(df)
    signals = compute_statistical_signals(df, baselines, graph, [])
    txn_c = "1c6db202-22d8-443f-86e7-fb1a8df05e84"
    # USER_C only has 1 transaction — std is 1.0 (default), z = (5000 - 5000) / 1 = 0
    assert signals[txn_c]["amount"] == pytest.approx(5000.0)
    assert "amount_zscore" in signals[txn_c]


def test_compute_statistical_signals_new_recipient(sample_df):
    from src.analysis import _coerce_types, compute_user_baselines, compute_graph_signals, compute_statistical_signals
    df = _coerce_types(sample_df)
    baselines = compute_user_baselines(df)
    graph = compute_graph_signals(df)
    signals = compute_statistical_signals(df, baselines, graph, [])
    # USER_A has transacted with USER_B before — not a new recipient on 2nd txn
    txn_b = "8830a720-ff34-4dce-a578-e5b8006b2976"
    assert signals[txn_b]["new_recipient"] is False


def test_format_txn_summary_contains_flags(sample_df):
    from src.analysis import (
        _coerce_types, compute_user_baselines, compute_graph_signals,
        compute_statistical_signals, format_txn_summary,
    )
    df = _coerce_types(sample_df)
    baselines = compute_user_baselines(df)
    graph = compute_graph_signals(df)
    signals = compute_statistical_signals(df, baselines, graph, [])
    txn_c = "1c6db202-22d8-443f-86e7-fb1a8df05e84"
    summary = format_txn_summary(txn_c, signals, baselines)
    assert "1c6db202" in summary  # UUID prefix shown
    assert "USER_C" in summary
    assert "SHARED_IBAN" in summary  # shared_iban_count == 2 triggers this flag
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_analysis.py::test_compute_statistical_signals_amount_zscore -v
```

Expected: `ImportError` — functions not yet defined.

- [ ] **Step 3: Append `compute_statistical_signals` and `format_txn_summary` to `src/analysis.py`**

```python
def load_locations(data_dir: Path) -> list[dict]:
    """Load locations.json as a list of dicts.

    Args:
        data_dir: Path to the dataset directory.

    Returns:
        List of location ping dicts with keys: biotag, timestamp, lat, lng, city.
    """
    loc_path = data_dir / "locations.json"
    if not loc_path.exists():
        return []
    with open(loc_path) as f:
        return json.load(f)


def compute_statistical_signals(
    df: pd.DataFrame,
    baselines: dict[str, dict],
    graph: dict[str, dict],
    locations: list[dict],
) -> dict[str, dict]:
    """Compute per-transaction anomaly signals relative to user baselines.

    Args:
        df: Transactions DataFrame (type-coerced).
        baselines: Output of compute_user_baselines().
        graph: Output of compute_graph_signals().
        locations: List of GPS ping dicts from locations.json.

    Returns:
        Dict mapping transaction_id to signal dict.
    """
    loc_lookup = _build_location_lookup(locations, df)
    signals: dict[str, dict] = {}

    for _, row in df.iterrows():
        txn_id = str(row[COL_TXN_ID])
        user_id = str(row[COL_SENDER_ID])
        baseline = baselines.get(user_id, {})

        amount = float(row[COL_AMOUNT]) if pd.notna(row[COL_AMOUNT]) else 0.0
        mean_amt = baseline.get("mean_amount", amount)
        std_amt = baseline.get("std_amount", 1.0) or 1.0
        amount_zscore = round((amount - mean_amt) / std_amt, 2)

        txn_hour = int(row[COL_TIMESTAMP].hour)
        typical_hours = baseline.get("typical_hours", {})
        hour_deviation = typical_hours.get(txn_hour, 0) < 2

        typical_types = baseline.get("typical_types", {})
        type_shift = str(row[COL_TXN_TYPE]) not in typical_types

        known_recipients = baseline.get("known_recipients", set())
        new_recipient = str(row.get(COL_RECIPIENT_ID, "")) not in known_recipients

        balance = float(row[COL_BALANCE_AFTER]) if pd.notna(row[COL_BALANCE_AFTER]) else None
        balance_low = balance is not None and balance < mean_amt * 0.1

        loc_coherence: bool | None = None
        if str(row.get(COL_TXN_TYPE, "")) == "payment" and str(row.get(COL_LOCATION, "")):
            loc_coherence = _check_location_coherence(
                user_id, row[COL_TIMESTAMP], str(row[COL_LOCATION]), loc_lookup
            )

        graph_sig = graph.get(txn_id, {})
        signals[txn_id] = {
            "user_id": user_id,
            "amount": amount,
            "amount_zscore": amount_zscore,
            "hour": txn_hour,
            "hour_deviation": hour_deviation,
            "type_shift": type_shift,
            "new_recipient": new_recipient,
            "balance_after": balance,
            "balance_low": balance_low,
            "location_coherence": loc_coherence,
            "shared_iban_count": graph_sig.get("shared_iban_count", 1),
            "circular_transfer": graph_sig.get("circular_transfer", False),
            "txn_type": str(row[COL_TXN_TYPE]),
            "recipient_id": str(row.get(COL_RECIPIENT_ID, "")),
            "recipient_iban": str(row.get(COL_RECIPIENT_IBAN, "")),
            "description": str(row.get(COL_DESCRIPTION, "")),
            "timestamp": str(row[COL_TIMESTAMP]),
        }
    return signals


def _build_location_lookup(
    locations: list[dict],
    df: pd.DataFrame,
) -> dict[str, list[dict]]:
    """Map sender_id → list of GPS pings (matched via BioTag prefix)."""
    lookup: dict[str, list[dict]] = {}
    for loc in locations:
        biotag = str(loc.get("biotag", ""))
        user_id = _biotag_to_sender_id(biotag, df)
        if user_id:
            lookup.setdefault(user_id, []).append({
                "ts": pd.to_datetime(loc["timestamp"]),
                "city": str(loc.get("city", "")),
            })
    return lookup


def _biotag_to_sender_id(biotag: str, df: pd.DataFrame) -> str | None:
    """Match a BioTag to a sender_id via exact or prefix match."""
    if biotag in df[COL_SENDER_ID].values:
        return biotag
    prefix = "-".join(biotag.split("-")[:3])
    if not prefix:
        return None
    matches = df[df[COL_SENDER_ID].str.startswith(prefix, na=False)][COL_SENDER_ID]
    return str(matches.iloc[0]) if not matches.empty else None


def _check_location_coherence(
    user_id: str,
    txn_time: pd.Timestamp,
    txn_location: str,
    loc_lookup: dict[str, list[dict]],
) -> bool:
    """Check if a GPS ping exists near the transaction time and location.

    Args:
        user_id: Sender's ID.
        txn_time: Transaction timestamp.
        txn_location: City or location string from transaction.
        loc_lookup: Output of _build_location_lookup().

    Returns:
        True if a coherent GPS ping is found, False otherwise.
    """
    pings = loc_lookup.get(user_id, [])
    window_start = txn_time - pd.Timedelta(hours=2)
    window_end = txn_time + pd.Timedelta(hours=2)
    for ping in pings:
        if window_start <= ping["ts"] <= window_end:
            if txn_location.lower() in ping["city"].lower():
                return True
    return False


def format_txn_summary(
    txn_id: str,
    signals: dict[str, dict],
    baselines: dict[str, dict],
) -> str:
    """Format a transaction's signals into a compact text block for LLM input.

    Args:
        txn_id: Full UUID of the transaction.
        signals: Output of compute_statistical_signals().
        baselines: Output of compute_user_baselines().

    Returns:
        ~10-line text summary including all risk flags.
    """
    s = signals[txn_id]
    user_id = s["user_id"]
    b = baselines.get(user_id, {})

    flags = []
    if abs(s["amount_zscore"]) > 2:
        flags.append(f"AMOUNT_ANOMALY(z={s['amount_zscore']:+.1f})")
    if s["hour_deviation"]:
        flags.append(f"UNUSUAL_HOUR({s['hour']}h)")
    if s["type_shift"]:
        flags.append(f"NEW_TXN_TYPE({s['txn_type']})")
    if s["new_recipient"]:
        flags.append("NEW_RECIPIENT")
    if s["balance_low"]:
        flags.append("BALANCE_CRITICALLY_LOW")
    if s["shared_iban_count"] > 2:
        flags.append(f"SHARED_IBAN({s['shared_iban_count']}_senders)")
    if s["circular_transfer"]:
        flags.append("CIRCULAR_TRANSFER")
    if s["location_coherence"] is False:
        flags.append("LOCATION_MISMATCH")

    flags_str = ", ".join(flags) if flags else "none"
    mean_amt = b.get("mean_amount", 0.0)

    return (
        f"TXN {txn_id[:8]}: user={user_id} (avg_txn=\u20ac{mean_amt:.0f})\n"
        f"  amount=\u20ac{s['amount']:.2f} (z={s['amount_zscore']:+.1f}), "
        f"type={s['txn_type']}, hour={s['hour']}h\n"
        f"  recipient={s['recipient_id'] or 'N/A'}, "
        f"balance_after=\u20ac{s['balance_after'] or 'N/A'}\n"
        f"  description=\"{str(s['description'])[:60]}\"\n"
        f"  FLAGS: {flags_str}"
    )
```

- [ ] **Step 4: Run all analysis tests**

```bash
python -m pytest tests/test_analysis.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/analysis.py tests/test_analysis.py
git commit -m "feat: per-transaction anomaly signals and text formatter"
```

---

## Task 4: Audio transcription (`src/transcription.py`)

**Files:**
- Create: `src/transcription.py`

No unit tests (requires real audio + API). Tested via integration in Task 11.

- [ ] **Step 1: Create `src/transcription.py`**

```python
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
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
python -c "from src.transcription import transcribe_audio_files; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/transcription.py
git commit -m "feat: audio transcription with disk cache via Gemini Flash Lite"
```

---

## Task 5: Communications Agent (`src/sub_agents.py`)

**Files:**
- Modify: `src/sub_agents.py`

- [ ] **Step 1: Rewrite `src/sub_agents.py` with the Communications Agent**

```python
"""
Specialist sub-agents: Communications Agent and Transaction Agent.
Both are called as tools by the orchestrator — see orchestrator.py.
"""

import logging

from strands import Agent

from config import MODEL_COMMS_AGENT, MODEL_TXN_AGENT
from src.utils import make_model, call_with_retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Communications Agent
# ---------------------------------------------------------------------------

_COMMS_PROMPT = """You are a Communications Intelligence Analyst specialising in fraud detection.
You examine SMS threads, email correspondence, and audio transcripts to build a risk profile for a citizen.

Look for:
- Social engineering: urgency, pressure to transfer money, promises of rewards
- Impersonation: messages pretending to be from a bank, government, or employer
- Coercion: threats, blackmail, or manipulation
- Fraud recruitment: offers to receive/forward money for a cut
- Suspicious links, unusual phone numbers, or requests for account credentials

Output your assessment in EXACTLY this format:
USER_RISK_PROFILE: <user_id>
RISK_LEVEL: low/medium/high
RED_FLAGS: <comma-separated list of specific signals found, or "none">
VICTIM_INDICATORS: <evidence the user may be deceived or coerced, or "none">
PERPETRATOR_INDICATORS: <evidence the user may be knowingly participating in fraud, or "none">
SUMMARY: <one paragraph narrative assessment>"""


def run_comms_agent(
    user_id: str,
    user_profile: dict,
    sms_threads: list[dict],
    email_threads: list[dict],
    audio_transcripts: dict[str, str],
    model_id: str = MODEL_COMMS_AGENT,
) -> str:
    """Analyse all communications for one user and return a risk profile.

    Args:
        user_id: The user's sender_id.
        user_profile: Dict with keys: first_name, last_name, birth_year,
            salary, job, iban, residence, description.
        sms_threads: List of SMS dicts with key 'sms'.
        email_threads: List of email dicts with key 'mail'.
        audio_transcripts: Dict mapping speaker_name to transcript text.
        model_id: Model to use.

    Returns:
        Structured USER_RISK_PROFILE text block.
    """
    agent = Agent(
        model=make_model(model_id),
        system_prompt=_COMMS_PROMPT,
        callback_handler=None,
        name=f"comms_agent_{user_id}",
    )

    speaker_name = (
        f"{user_profile.get('first_name', '')}_{user_profile.get('last_name', '')}"
        .lower()
        .replace(" ", "_")
    )
    transcript_text = audio_transcripts.get(speaker_name, "none")

    sms_text = "\n---\n".join(s.get("sms", "") for s in sms_threads) or "none"
    email_text = "\n---\n".join(e.get("mail", "") for e in email_threads) or "none"

    prompt = (
        f"USER ID: {user_id}\n"
        f"PROFILE: {user_profile.get('description', 'No description available.')}\n\n"
        f"=== SMS THREADS ===\n{sms_text}\n\n"
        f"=== EMAIL THREADS ===\n{email_text}\n\n"
        f"=== AUDIO TRANSCRIPTS ===\n{transcript_text}\n\n"
        f"Provide the risk profile for user {user_id}."
    )

    result = call_with_retry(
        lambda: str(agent(prompt)),
        label=f"comms_agent_{user_id}",
    )
    return result


# ---------------------------------------------------------------------------
# Transaction Agent
# ---------------------------------------------------------------------------

_TXN_PROMPT = """You are a Financial Pattern Analyst specialising in fraud detection.
You receive pre-computed statistical signal summaries for a batch of transactions.
Each summary includes the user's baseline and per-transaction anomaly flags.

Your task: for each transaction, reason about whether the deviations from baseline
are consistent with a legitimate explanation given the user's profile, or indicative of fraud.

Consider:
- Is the amount unusual relative to this specific user's history? A z-score of +2 for a
  high-salary professional is very different from a pensioner.
- Is the time-of-day suspicious for this user's known habits?
- Does the recipient or transaction type represent a genuine first-time event, or
  does the pattern suggest account takeover or social engineering?
- Do combinations of flags amplify risk (e.g. new recipient + unusual hour + high amount)?

Output EXACTLY one line per transaction:
<full-uuid>: RISK_SCORE (0.0–1.0) | RATIONALE (one sentence)

Score guide: 0.0–0.2 routine, 0.3–0.5 mild anomaly, 0.6–0.8 suspicious, 0.9–1.0 strongly fraudulent.
Output ALL transactions in the batch. No preamble, no summary."""


def run_transaction_agent(
    txn_summaries: list[str],
    model_id: str = MODEL_TXN_AGENT,
) -> str:
    """Run the Transaction Agent on a batch of pre-formatted summaries.

    Args:
        txn_summaries: List of text summaries, one per transaction.
            Each produced by analysis.format_txn_summary().
        model_id: Model to use.

    Returns:
        Raw text with one 'UUID: SCORE | RATIONALE' line per transaction.
    """
    agent = Agent(
        model=make_model(model_id),
        system_prompt=_TXN_PROMPT,
        callback_handler=None,
        name="transaction_agent",
    )
    batch_text = "\n\n".join(txn_summaries)
    result = call_with_retry(
        lambda: str(agent(f"Analyse these transactions:\n\n{batch_text}")),
        label="transaction_agent",
    )
    return result
```

- [ ] **Step 2: Verify imports**

```bash
python -c "from src.sub_agents import run_comms_agent, run_transaction_agent; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/sub_agents.py
git commit -m "feat: Communications Agent and Transaction Agent with structured prompts"
```

---

## Task 6: Network Analysis (`src/network.py`)

**Files:**
- Create: `src/network.py`

- [ ] **Step 1: Create `src/network.py`**

```python
"""
Network Analysis Tool — dataset-level recipient IBAN pattern reasoning.
Called by the orchestrator as a @tool when a suspicious IBAN is flagged.
"""

import logging

from strands import Agent

from config import MODEL_TXN_AGENT
from src.utils import make_model, call_with_retry

logger = logging.getLogger(__name__)

_NETWORK_PROMPT = """You are a Financial Network Intelligence Analyst.
You receive a summary of all transactions flowing to a specific recipient IBAN,
including how many distinct senders are involved and their profiles.

Your task: determine whether this IBAN looks like:
- A normal merchant or service provider receiving payments from many customers
- A salary/payroll account receiving from employers
- A suspicious collection point receiving from unrelated individuals (mule account)
- A circular transfer intermediary

Output in this exact format:
RECIPIENT_IBAN: <iban>
NETWORK_CLASS: merchant/payroll/mule/circular/unknown
CONFIDENCE: high/medium/low
EVIDENCE: <one sentence describing the key pattern>
RISK_SIGNAL: <0.0-1.0 risk score for this recipient being involved in fraud>"""


def get_network_context(
    recipient_iban: str,
    all_transactions: list[dict],
    model_id: str = MODEL_TXN_AGENT,
) -> str:
    """Analyse all transactions sharing a recipient IBAN across the dataset.

    Args:
        recipient_iban: The IBAN to investigate.
        all_transactions: List of all transaction dicts (raw rows from CSV).
        model_id: Model to use.

    Returns:
        Structured RECIPIENT_IBAN analysis text block.
    """
    related = [
        t for t in all_transactions
        if str(t.get("recipient_iban", "")) == recipient_iban
    ]

    if not related:
        return f"RECIPIENT_IBAN: {recipient_iban}\nNETWORK_CLASS: unknown\nCONFIDENCE: low\nEVIDENCE: No transactions found.\nRISK_SIGNAL: 0.0"

    senders = {str(t.get("sender_id", "")) for t in related}
    total_amount = sum(float(t.get("amount", 0) or 0) for t in related)
    descriptions = list({str(t.get("description", "")) for t in related if t.get("description")})[:5]

    summary = (
        f"Recipient IBAN: {recipient_iban}\n"
        f"Total transactions: {len(related)}\n"
        f"Distinct senders: {len(senders)} — {', '.join(sorted(senders)[:10])}\n"
        f"Total amount received: €{total_amount:.2f}\n"
        f"Sample descriptions: {'; '.join(descriptions) or 'none'}"
    )

    agent = Agent(
        model=make_model(model_id),
        system_prompt=_NETWORK_PROMPT,
        callback_handler=None,
        name="network_agent",
    )
    result = call_with_retry(
        lambda: str(agent(f"Analyse this recipient network:\n\n{summary}")),
        label=f"network_agent_{recipient_iban[:8]}",
    )
    return result
```

- [ ] **Step 2: Verify imports**

```bash
python -c "from src.network import get_network_context; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/network.py
git commit -m "feat: network analysis tool for dataset-level IBAN graph reasoning"
```

---

## Task 7: Dynamic Orchestrator (`src/orchestrator.py`)

**Files:**
- Modify: `src/orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_orchestrator.py
from src.orchestrator import parse_classifications

TXN_A = "e1021ab7-c2de-4791-994b-bab86e6fbe3e"
TXN_B = "8830a720-ff34-4dce-a578-e5b8006b2976"


def test_parse_classifications_fraud():
    output = f"{TXN_A}: 1 | high | Suspicious transfer at 3am to new recipient"
    result = parse_classifications(output)
    assert result == {TXN_A: 1}


def test_parse_classifications_legitimate():
    output = f"{TXN_B}: 0 | low | Normal salary payment"
    result = parse_classifications(output)
    assert result == {TXN_B: 0}


def test_parse_classifications_mixed_batch():
    output = (
        f"{TXN_A}: 1 | high | Circular transfer detected\n"
        f"{TXN_B}: 0 | medium | Slightly unusual hour but known recipient"
    )
    result = parse_classifications(output)
    assert result == {TXN_A: 1, TXN_B: 0}


def test_parse_classifications_case_insensitive():
    output = f"{TXN_A.upper()}: 1 | high | reason"
    result = parse_classifications(output)
    assert result == {TXN_A: 1}


def test_parse_classifications_empty():
    assert parse_classifications("No output here") == {}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_orchestrator.py -v
```

Expected: some pass (parse_classifications already partially updated), but UUID lowercasing may differ. Verify the exact failures.

- [ ] **Step 3: Rewrite `src/orchestrator.py`**

```python
"""
Dynamic Orchestrator — the directing intelligence of the pipeline.

A strands Agent that receives statistical signals and cached user profiles,
then selectively calls @tool functions to gather additional evidence before
making final binary fraud classifications.
"""

import logging
import re
from pathlib import Path

from strands import Agent, tool

from config import MODEL_ORCHESTRATOR, MODEL_TXN_AGENT, BATCH_SIZE_TXN_AGENT
from src.utils import make_model, call_with_retry, validate_output, UUID_PATTERN
from src.analysis import format_txn_summary

logger = logging.getLogger(__name__)

_ORCHESTRATOR_PROMPT = """You are the Chief Fraud Intelligence Officer at MirrorPay.

You receive statistical anomaly signals and user intelligence profiles for a batch of transactions.
Your job is to make the final binary fraud classification for each transaction.

TOOLS AVAILABLE — call them when you need more evidence:
- analyze_financial_patterns(txn_ids, user_id): Deep LLM analysis of specific transactions
- analyze_recipient_network(recipient_iban): Dataset-level network analysis of a recipient IBAN
- get_location_check(user_id, txn_id, timestamp, location): GPS coherence check
- get_comms_detail(user_id): Full communications risk profile for a user

DECISION PRINCIPLES:
1. ECONOMIC WEIGHTING (most important): Scale your intervention threshold by amount.
   A €50,000 uncertain transfer → classify as fraud (1). A €5 uncertain coffee → classify as legitimate (0).
   Missing a high-value fraud costs far more than a false positive on a small transaction.

2. CONVERGENCE: Multiple weak signals agreeing (unusual hour + new recipient + high amount) = strong fraud signal.
   A single anomaly flag with no supporting evidence = likely legitimate.

3. PROFILE-RELATIVE reasoning: An 86-year-old pensioner wiring €10,000 to a new foreign IBAN at 3am
   is very different from a 30-year-old professional doing the same. Always reason relative to the user.

4. THREAT INTELLIGENCE: If prior-level patterns are provided, actively look for evolved versions
   of those tactics. Fraudsters adapt but rarely completely reinvent their methods.

5. INTERVENTION BIAS: When genuinely uncertain after investigating, classify as fraud (1).
   False negatives (missed fraud) carry higher regulatory and economic cost.

OUTPUT FORMAT — output EXACTLY one line per transaction, ALL transactions:
<full-uuid>: CLASSIFICATION (0 or 1) | CONFIDENCE (high/medium/low) | REASON (one sentence)

No preamble. No summary. Just the classification lines."""


def make_orchestrator_agent(
    signals: dict[str, dict],
    baselines: dict[str, dict],
    user_profiles: dict[str, str],
    all_transactions: list[dict],
    threat_intel: str | None,
    model_id: str = MODEL_ORCHESTRATOR,
) -> Agent:
    """Build the dynamic orchestrator agent with all tools wired via closures.

    Args:
        signals: Output of compute_statistical_signals().
        baselines: Output of compute_user_baselines().
        user_profiles: Dict mapping user_id to comms risk profile text.
        all_transactions: Raw transaction dicts (list of CSV rows as dicts).
        threat_intel: Threat intelligence brief from previous level, or None.
        model_id: Orchestrator model to use.

    Returns:
        Configured strands Agent ready to classify a batch of transactions.
    """
    from src.sub_agents import run_transaction_agent
    from src.network import get_network_context

    @tool
    def analyze_financial_patterns(txn_ids: list[str], user_id: str) -> str:
        """Run deep financial pattern analysis on specific transactions.

        Args:
            txn_ids: List of transaction UUIDs to analyse.
            user_id: The sender's ID for baseline context.

        Returns:
            Per-transaction risk scores and rationales.
        """
        summaries = [
            format_txn_summary(tid, signals, baselines)
            for tid in txn_ids
            if tid in signals
        ]
        if not summaries:
            return "No signal data found for the requested transactions."
        return run_transaction_agent(summaries, model_id=MODEL_TXN_AGENT)

    @tool
    def analyze_recipient_network(recipient_iban: str) -> str:
        """Analyse all transactions in the dataset sharing this recipient IBAN.

        Args:
            recipient_iban: The IBAN to investigate for mule/merchant patterns.

        Returns:
            Network classification with evidence and risk score.
        """
        return get_network_context(
            recipient_iban, all_transactions, model_id=MODEL_TXN_AGENT
        )

    @tool
    def get_location_check(
        user_id: str, txn_id: str, timestamp: str, location: str
    ) -> str:
        """Check if GPS data supports the user being near a transaction location.

        Args:
            user_id: The sender's ID.
            txn_id: The transaction UUID.
            timestamp: Transaction timestamp string.
            location: Location/city from the transaction.

        Returns:
            'COHERENT', 'INCOHERENT', or 'NO_DATA'.
        """
        sig = signals.get(txn_id, {})
        coherence = sig.get("location_coherence")
        if coherence is None:
            return f"NO_DATA: No GPS pings found for user {user_id} around {timestamp}."
        return f"{'COHERENT' if coherence else 'INCOHERENT'}: GPS data {'supports' if coherence else 'contradicts'} user presence near {location} at {timestamp}."

    @tool
    def get_comms_detail(user_id: str) -> str:
        """Retrieve the Communications Agent risk profile for a user.

        Args:
            user_id: The user's sender_id.

        Returns:
            Full communications risk profile text.
        """
        return user_profiles.get(
            user_id,
            f"USER_RISK_PROFILE: {user_id}\nRISK_LEVEL: unknown\nRED_FLAGS: none\nSUMMARY: No communications data available.",
        )

    system_prompt = _ORCHESTRATOR_PROMPT
    if threat_intel:
        system_prompt = (
            f"THREAT INTELLIGENCE FROM PREVIOUS LEVEL:\n{threat_intel}\n\n"
            "---\n\n" + system_prompt
        )

    return Agent(
        model=make_model(model_id),
        system_prompt=system_prompt,
        tools=[
            analyze_financial_patterns,
            analyze_recipient_network,
            get_location_check,
            get_comms_detail,
        ],
        callback_handler=None,
        name="orchestrator",
    )


def run_orchestrator_batch(
    txn_ids: list[str],
    signals: dict[str, dict],
    baselines: dict[str, dict],
    user_profiles: dict[str, str],
    all_transactions: list[dict],
    threat_intel: str | None,
    model_id: str = MODEL_ORCHESTRATOR,
) -> dict[str, int]:
    """Classify one batch of transactions using the dynamic orchestrator.

    Args:
        txn_ids: Transaction UUIDs in this batch.
        signals: Full signals dict (orchestrator only reads relevant entries).
        baselines: Full baselines dict.
        user_profiles: Full user profiles dict.
        all_transactions: All raw transaction dicts.
        threat_intel: Prior-level threat intelligence brief, or None.
        model_id: Orchestrator model.

    Returns:
        Dict mapping transaction_id to classification (0 or 1).
    """
    agent = make_orchestrator_agent(
        signals, baselines, user_profiles, all_transactions, threat_intel, model_id
    )

    batch_lines = []
    for tid in txn_ids:
        if tid in signals:
            batch_lines.append(format_txn_summary(tid, signals, baselines))
            batch_lines.append(
                f"  user_profile_risk: {user_profiles.get(signals[tid]['user_id'], 'unknown')[:100]}"
            )
    batch_text = "\n\n".join(batch_lines)

    raw = call_with_retry(
        lambda: str(agent(
            f"Classify these {len(txn_ids)} transactions. "
            f"Use tools when you need more evidence.\n\n{batch_text}"
        )),
        label="orchestrator_batch",
    )
    return parse_classifications(raw)


def parse_classifications(orchestrator_output: str) -> dict[str, int]:
    """Extract transaction UUIDs and binary classifications from orchestrator output.

    Args:
        orchestrator_output: Raw text output from the orchestrator agent.

    Returns:
        Dict mapping transaction UUID (lowercased) to classification (0 or 1).
    """
    pattern = re.compile(
        r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
        r":\s*([01])\b",
        re.IGNORECASE,
    )
    matches = pattern.findall(orchestrator_output)
    return {txn_id.lower(): int(cls) for txn_id, cls in matches}
```

- [ ] **Step 4: Run orchestrator tests**

```bash
python -m pytest tests/test_orchestrator.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: dynamic orchestrator with @tool functions and parse_classifications"
```

---

## Task 8: Memory Agent (`src/memory.py`)

**Files:**
- Create: `src/memory.py`
- Create: `tests/test_memory.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_memory.py
import json
from pathlib import Path
from src.memory import load_threat_intel


def test_load_threat_intel_returns_none_when_missing(tmp_path):
    result = load_threat_intel(1, memory_dir=tmp_path)
    assert result is None


def test_load_threat_intel_returns_content_when_present(tmp_path):
    brief = "THREAT_INTEL: Level 1 → Level 2\nFRAUD_PATTERNS_FOUND: smishing"
    (tmp_path / "level_1_intel.md").write_text(brief)
    result = load_threat_intel(1, memory_dir=tmp_path)
    assert result == brief


def test_load_threat_intel_reads_correct_level(tmp_path):
    (tmp_path / "level_1_intel.md").write_text("level 1 intel")
    (tmp_path / "level_2_intel.md").write_text("level 2 intel")
    assert load_threat_intel(1, memory_dir=tmp_path) == "level 1 intel"
    assert load_threat_intel(2, memory_dir=tmp_path) == "level 2 intel"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_memory.py -v
```

Expected: `ImportError` — module not yet created.

- [ ] **Step 3: Create `src/memory.py`**

```python
"""
Memory Agent — synthesises fraud patterns after each level into a
threat intelligence brief loaded by the orchestrator at the next level.
"""

import logging
from pathlib import Path

from strands import Agent

from config import MODEL_MEMORY_AGENT, MEMORY_DIR
from src.utils import make_model, call_with_retry

logger = logging.getLogger(__name__)

_MEMORY_PROMPT = """You are a Threat Intelligence Analyst synthesising learnings from a completed fraud detection exercise.

You will receive:
- A list of transactions classified as fraudulent, with their orchestrator reasoning
- Statistical signals for those transactions
- User risk profiles involved

Your task: identify the recurring tactics, patterns, and signals that distinguished fraud
from legitimate transactions. Write a threat intelligence brief that will help the next
level's investigator anticipate evolved versions of these tactics.

Output in EXACTLY this format:
THREAT_INTEL: Level <N> → Level <N+1>
FRAUD_PATTERNS_FOUND: <comma-separated list of observed tactics>
HIGH_RISK_SIGNALS: <which statistical signals were most predictive>
VICTIM_PROFILE: <what types of users were targeted>
PERPETRATOR_PROFILE: <what types of users initiated fraud, if any>
EVOLVING_TACTICS_TO_WATCH: <what the next level's investigator should look for>
NARRATIVE: <one paragraph synthesis>"""


def run_memory_agent(
    completed_level: int,
    fraud_classifications: dict[str, int],
    signals: dict[str, dict],
    user_profiles: dict[str, str],
    memory_dir: Path | None = None,
    model_id: str = MODEL_MEMORY_AGENT,
) -> str:
    """Synthesise fraud patterns from a completed level into a threat brief.

    Args:
        completed_level: The level number just completed (1, 2, ...).
        fraud_classifications: Full classification dict {txn_id: 0_or_1}.
        signals: Statistical signals dict from compute_statistical_signals().
        user_profiles: Communications risk profiles dict {user_id: profile_text}.
        memory_dir: Directory to save the brief. Defaults to config.MEMORY_DIR.
        model_id: Model to use.

    Returns:
        The threat intelligence brief text.
    """
    mem_dir = Path(memory_dir or MEMORY_DIR)
    mem_dir.mkdir(parents=True, exist_ok=True)

    fraud_txn_ids = [tid for tid, cls in fraud_classifications.items() if cls == 1]
    if not fraud_txn_ids:
        brief = (
            f"THREAT_INTEL: Level {completed_level} → Level {completed_level + 1}\n"
            "FRAUD_PATTERNS_FOUND: none detected\n"
            "HIGH_RISK_SIGNALS: none\nVICTIM_PROFILE: unknown\n"
            "PERPETRATOR_PROFILE: unknown\n"
            "EVOLVING_TACTICS_TO_WATCH: none\n"
            "NARRATIVE: No fraudulent transactions were identified in this level."
        )
    else:
        fraud_summaries = []
        seen_users: set[str] = set()
        for tid in fraud_txn_ids[:30]:  # cap to avoid context overflow
            sig = signals.get(tid, {})
            user_id = sig.get("user_id", "unknown")
            flags = [
                k for k, v in sig.items()
                if v is True or (isinstance(v, (int, float)) and k.endswith("_count") and v > 2)
            ]
            fraud_summaries.append(
                f"TXN {tid[:8]}: user={user_id}, amount=€{sig.get('amount', 0):.0f}, "
                f"flags={flags}"
            )
            if user_id not in seen_users:
                seen_users.add(user_id)

        profile_snippets = "\n".join(
            f"--- {uid} ---\n{user_profiles.get(uid, 'no profile')[:300]}"
            for uid in list(seen_users)[:5]
        )

        prompt = (
            f"Completed level: {completed_level}\n"
            f"Fraudulent transactions detected ({len(fraud_txn_ids)} total, showing up to 30):\n"
            + "\n".join(fraud_summaries)
            + f"\n\nUser profiles involved:\n{profile_snippets}\n\n"
            "Write the threat intelligence brief."
        )

        agent = Agent(
            model=make_model(model_id),
            system_prompt=_MEMORY_PROMPT,
            callback_handler=None,
            name="memory_agent",
        )
        brief = call_with_retry(
            lambda: str(agent(prompt)),
            label="memory_agent",
        )

    out_path = mem_dir / f"level_{completed_level}_intel.md"
    out_path.write_text(brief, encoding="utf-8")
    logger.info("Threat intelligence saved to %s", out_path)
    return brief


def load_threat_intel(
    completed_level: int,
    memory_dir: Path | None = None,
) -> str | None:
    """Load threat intelligence brief produced after level N.

    Args:
        completed_level: The level whose intel brief to load.
        memory_dir: Directory containing brief files. Defaults to config.MEMORY_DIR.

    Returns:
        Brief text, or None if not found.
    """
    mem_dir = Path(memory_dir or MEMORY_DIR)
    path = mem_dir / f"level_{completed_level}_intel.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")
```

- [ ] **Step 4: Run memory tests**

```bash
python -m pytest tests/test_memory.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/memory.py tests/test_memory.py
git commit -m "feat: Memory Agent synthesises cross-level threat intelligence"
```

---

## Task 9: Pipeline wiring (`main.py`)

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Rewrite `main.py`**

```python
"""Main entry point: runs the multi-agent fraud detection pipeline.

Usage:
    python main.py <level>        # run one level (1, 2, or 3)
    python main.py all            # run all available levels in sequence

Output:
    output/level_<N>_output.txt   # fraudulent transaction IDs, one per line
    Prints Langfuse session ID to stdout for submission.
"""

import json
import logging
import sys
import uuid
from pathlib import Path

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    LEVEL_FOLDERS,
    BATCH_SIZE_ORCHESTRATOR,
    MODEL_ORCHESTRATOR,
)
from src.analysis import (
    load_transactions,
    load_locations,
    compute_user_baselines,
    compute_graph_signals,
    compute_statistical_signals,
)
from src.memory import load_threat_intel, run_memory_agent
from src.orchestrator import run_orchestrator_batch, parse_classifications
from src.sub_agents import run_comms_agent
from src.transcription import transcribe_audio_files
from src.utils import (
    ContextOverflowError,
    make_langfuse_client,
    validate_output,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _batch(items: list, size: int) -> list[list]:
    return [items[i: i + size] for i in range(0, len(items), size)]


def run_level(level: int, session_id: str) -> dict[str, int]:
    """Run the full fraud detection pipeline for one dataset level.

    Args:
        level: Dataset level number (1–5).
        session_id: Langfuse session ID for tracing.

    Returns:
        Dict mapping transaction_id to classification (0=legit, 1=fraud).
    """
    folder_name = LEVEL_FOLDERS.get(level)
    if not folder_name:
        raise ValueError(f"Level {level} not in LEVEL_FOLDERS — add it to config.py")

    data_dir = Path(DATA_DIR) / folder_name
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    logger.info("=== Level %d: %s ===", level, folder_name)

    # -----------------------------------------------------------------------
    # Phase 0a: Statistical pre-processing (free)
    # -----------------------------------------------------------------------
    logger.info("Phase 0a: computing statistical signals...")
    df = load_transactions(data_dir)
    locations = load_locations(data_dir)
    baselines = compute_user_baselines(df)
    graph = compute_graph_signals(df)
    signals = compute_statistical_signals(df, baselines, graph, locations)
    all_txn_ids = list(df["transaction_id"].astype(str))
    all_transactions = df.to_dict("records")

    logger.info("Loaded %d transactions for %d users", len(all_txn_ids), len(baselines))

    # -----------------------------------------------------------------------
    # Phase 0b: Audio transcription (cached to disk)
    # -----------------------------------------------------------------------
    logger.info("Phase 0b: audio transcription...")
    audio_dir = data_dir / "audio"
    audio_transcripts = transcribe_audio_files(audio_dir)
    logger.info("Transcribed %d speakers", len(audio_transcripts))

    # -----------------------------------------------------------------------
    # Phase 1: Communications Agent — one call per user
    # -----------------------------------------------------------------------
    logger.info("Phase 1: Communications Agent (%d users)...", len(baselines))
    users_data = _load_json(data_dir / "users.json")
    sms_data = _load_json(data_dir / "sms.json")
    email_data = _load_json(data_dir / "mails.json")

    # Index SMS and emails by user first_name+last_name for matching
    user_profiles: dict[str, str] = {}
    for user in users_data:
        # Use sender_id matching via IBAN — users.json contains 'iban'
        user_iban = user.get("iban", "")
        matching_ids = df[df["sender_iban"] == user_iban]["sender_id"].unique()
        user_id = str(matching_ids[0]) if len(matching_ids) > 0 else None

        if not user_id:
            logger.warning("Could not match user %s %s to a sender_id",
                           user.get("first_name"), user.get("last_name"))
            continue

        profile = run_comms_agent(
            user_id=user_id,
            user_profile=user,
            sms_threads=sms_data,
            email_threads=email_data,
            audio_transcripts=audio_transcripts,
        )
        user_profiles[user_id] = profile
        logger.info("Comms profile for %s: %s", user_id, profile[:80].replace("\n", " "))

    # -----------------------------------------------------------------------
    # Load threat intelligence from previous level (if any)
    # -----------------------------------------------------------------------
    threat_intel = load_threat_intel(level - 1) if level > 1 else None
    if threat_intel:
        logger.info("Loaded threat intelligence from level %d", level - 1)

    # -----------------------------------------------------------------------
    # Phase 2: Dynamic Orchestrator — batched with halve-on-overflow
    # -----------------------------------------------------------------------
    logger.info("Phase 2: Orchestrator classifying %d transactions...", len(all_txn_ids))
    all_classifications: dict[str, int] = {}
    batch_size = BATCH_SIZE_ORCHESTRATOR

    pending = list(all_txn_ids)
    while pending:
        batch = pending[:batch_size]
        try:
            result = run_orchestrator_batch(
                txn_ids=batch,
                signals=signals,
                baselines=baselines,
                user_profiles=user_profiles,
                all_transactions=all_transactions,
                threat_intel=threat_intel,
            )
            all_classifications.update(result)

            # Retry any IDs the model dropped
            _found, missing = validate_output(
                "\n".join(f"{k}: {v}" for k, v in result.items()),
                set(batch),
            )
            if missing:
                logger.warning("Orchestrator dropped %d IDs — retrying as mini-batch", len(missing))
                mini_result = run_orchestrator_batch(
                    txn_ids=list(missing),
                    signals=signals,
                    baselines=baselines,
                    user_profiles=user_profiles,
                    all_transactions=all_transactions,
                    threat_intel=threat_intel,
                )
                all_classifications.update(mini_result)

            pending = pending[batch_size:]

        except ContextOverflowError:
            if batch_size <= 1:
                raise RuntimeError("Batch size of 1 still causes context overflow")
            batch_size = max(1, batch_size // 2)
            logger.warning("Context overflow — reducing batch size to %d", batch_size)
            # Do not advance pending — retry the same batch with smaller size

    logger.info("Classified %d/%d transactions", len(all_classifications), len(all_txn_ids))

    # -----------------------------------------------------------------------
    # Memory Agent — synthesise patterns for next level
    # -----------------------------------------------------------------------
    logger.info("Running Memory Agent...")
    run_memory_agent(
        completed_level=level,
        fraud_classifications=all_classifications,
        signals=signals,
        user_profiles=user_profiles,
    )

    return all_classifications


def _write_output(level: int, classifications: dict[str, int]) -> Path:
    """Write fraudulent transaction IDs to the submission file.

    Args:
        level: Dataset level number.
        classifications: Full classification dict.

    Returns:
        Path to the output file.
    """
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"level_{level}_output.txt"

    fraud_ids = [tid for tid, cls in classifications.items() if cls == 1]
    fraud_count = len(fraud_ids)
    total = len(classifications)

    if fraud_count == 0:
        logger.warning("WARNING: No fraudulent transactions found — output will be invalid!")
    if fraud_count == total:
        logger.warning("WARNING: All transactions flagged as fraud — output will be invalid!")
    if total > 0 and fraud_count / total < 0.05:
        logger.warning(
            "WARNING: Only %.1f%% of transactions flagged as fraud — may be too conservative",
            100 * fraud_count / total,
        )

    out_path.write_text("\n".join(fraud_ids) + "\n", encoding="ascii")
    logger.info("Wrote %d fraudulent IDs to %s", fraud_count, out_path)
    return out_path


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <level|all>")
        sys.exit(1)

    arg = sys.argv[1].strip().lower()
    levels = list(LEVEL_FOLDERS.keys()) if arg == "all" else [int(arg)]

    session_id = str(uuid.uuid4())
    logger.info("Langfuse session ID: %s", session_id)
    print(f"\nLangfuse Session ID: {session_id}\n")

    for level in levels:
        classifications = run_level(level, session_id)
        _write_output(level, classifications)

    print(f"\n✓ Done. Session ID for submission: {session_id}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
python -c "from main import run_level, main; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: full pipeline wiring — run_level, batch orchestration, output writing"
```

---

## Task 10: Smoke test on Level 1

Run the full pipeline on the smallest dataset to verify end-to-end function before spending tokens on larger levels.

- [ ] **Step 1: Check `.env` has required keys**

```bash
grep -E "OPENROUTER_API_KEY|LANGFUSE_PUBLIC_KEY|LANGFUSE_SECRET_KEY|LANGFUSE_HOST" .env
```

Expected: all four keys present and non-empty.

- [ ] **Step 2: Run Level 1 (The Truman Show — 80 transactions)**

```bash
python main.py 1
```

Expected output:
```
Langfuse Session ID: <uuid>
INFO: === Level 1: The Truman Show - train ===
INFO: Phase 0a: computing statistical signals...
INFO: Loaded 80 transactions for N users
INFO: Phase 0b: audio transcription...
INFO: Phase 1: Communications Agent (N users)...
INFO: Phase 2: Orchestrator classifying 80 transactions...
INFO: Classified 80/80 transactions
INFO: Wrote K fraudulent IDs to output/level_1_output.txt
✓ Done. Session ID for submission: <uuid>
```

- [ ] **Step 3: Verify output is valid**

```bash
wc -l output/level_1_output.txt
python -c "
lines = open('output/level_1_output.txt').read().strip().split()
total = 80
pct = 100 * len(lines) / total
print(f'Flagged {len(lines)}/{total} transactions ({pct:.1f}%)')
assert 0 < len(lines) < total, 'Output invalid: empty or all transactions flagged'
assert pct >= 5, f'Only {pct:.1f}% flagged — likely too conservative'
print('Output looks valid')
"
```

Expected: between 5–40% of transactions flagged, no assertion errors.

- [ ] **Step 4: Check Langfuse for the session**

Log into Langfuse and verify the session ID shows traces for the comms agent, orchestrator, and memory agent calls.

- [ ] **Step 5: Commit**

```bash
git add output/.gitkeep memory/.gitkeep 2>/dev/null || true
git commit -m "test: smoke test Level 1 passes end-to-end"
```

---

## Task 11: Add `pytest` and update `requirements.txt`

- [ ] **Step 1: Add test dependencies**

```bash
echo "pytest" >> requirements.txt
pip install pytest
```

- [ ] **Step 2: Run all unit tests together**

```bash
python -m pytest tests/ -v --ignore=tests/integration
```

Expected: all unit tests PASS (no API calls made).

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add pytest to requirements"
```

---

## Self-Review

Spec coverage check:

| Spec section | Implemented in |
|---|---|
| Phase 0a: Statistical signals | Task 2, Task 3 |
| Phase 0b: Audio transcription | Task 4 |
| Communications Agent (per user) | Task 5 |
| Transaction Agent (on-demand tool) | Task 5 |
| Network Analysis Tool | Task 6 |
| Dynamic Orchestrator with tools | Task 7 |
| Memory Agent + load_threat_intel | Task 8 |
| Economic weighting in prompt | Task 7 (system prompt) |
| call_with_retry / validate_output | Task 1 |
| parse_classifications (UUID regex) | Task 7 |
| Batch size halving on 400 | Task 9 |
| Missing ID mini-batch retry | Task 9 |
| Output format (ASCII, one UUID/line) | Task 9 |
| LEVEL_FOLDERS mapping | config.py (done) |
| model_id per-agent | config.py (done) |
| Memory dir creation | Task 8 |
| Langfuse session ID to stdout | Task 9 |

No gaps found.
