"""
Specialist sub-agents: Communications Agent and Transaction Agent.
Both are called as tools by the orchestrator — see orchestrator.py.
"""

import logging
from datetime import datetime, timedelta

from strands import Agent

from config import (
    BATCH_SIZE_COMMS_AGENT,
    COMMS_WINDOW_DAYS_AFTER,
    COMMS_WINDOW_DAYS_BEFORE,
    MODEL_COMMS_AGENT,
    MODEL_TXN_AGENT,
)
from src.comms_filter import filter_emails_by_window, filter_sms_by_window
from src.utils import ContextOverflowError, call_agent_with_retry, make_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Communications Agent
# ---------------------------------------------------------------------------

_COMMS_PROMPT = """You are a Communications Intelligence Analyst specialising in fraud detection.
You examine SMS threads, email correspondence, and audio transcripts for ONE USER.

Your goal is to assess fraud risk PER TRANSACTION by identifying suspicious communications
that occurred BEFORE each transaction — social engineering or manipulation that may have
influenced the user to make that transaction.

Suspicious signals to detect:
- Social engineering: urgency, pressure to transfer money, promises of rewards
- Fake/lookalike domains with digit substitutions (paypa1-secure.net, amaz0n-verify.com, etc.)
- Impersonation: messages pretending to be from a bank, government, or employer
- Coercion: threats, blackmail, manipulation
- Fraud recruitment: offers to receive/forward money for a cut
- Suspicious shortened links (bit.ly, tinyurl) to unknown destinations
- Requests for OTPs, account credentials, or personal data

LEGITIMATE DOMAIN REFERENCE — these senders are NOT suspicious:
{legit_domains_sample}

For EACH TRANSACTION listed, output EXACTLY this block (no preamble, no extra text):
TXN_COMMS_RISK: <full-uuid>
PRE_TXN_FLAGS: <yes/no> | <brief description or "none">
DAYS_BEFORE: <number or N/A>
RISK_LEVEL: low/medium/high
REASON: <one sentence>
---"""


def _parse_txn_timestamp(raw: str) -> datetime | None:
    """Best-effort parse of a transaction timestamp string."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "").strip())
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(str(raw).strip(), fmt)
            except ValueError:
                continue
    return None


def _txn_batch_window(
    txn_batch: list[dict],
    days_before: int,
    days_after: int,
) -> tuple[datetime, datetime]:
    """Compute [earliest-before, latest+after] window across a transaction batch."""
    timestamps = [_parse_txn_timestamp(t.get("timestamp", "")) for t in txn_batch]
    timestamps = [ts for ts in timestamps if ts is not None]
    if not timestamps:
        return datetime.min, datetime.max
    return (
        min(timestamps) - timedelta(days=days_before),
        max(timestamps) + timedelta(days=days_after),
    )


def _build_comms_prompt(
    user_id: str,
    user_profile: dict,
    txn_batch: list[dict],
    sms_win: list[dict],
    email_win: list[dict],
    transcript_text: str,
) -> str:
    """Compose the user prompt for one comms-agent batch call."""
    name = f"{user_profile.get('first_name', '')} {user_profile.get('last_name', '')}"
    txn_list = "\n".join(
        f"  - {t['txn_id']} | {t['timestamp']} | "
        f"\u20ac{float(t.get('amount', 0)):.2f} | {t.get('txn_type', '?')} | "
        f"{str(t.get('description', ''))[:50]}"
        for t in txn_batch
    ) or "  (no transactions)"
    sms_text = "\n---\n".join(s.get("sms", "") for s in sms_win) or "none"
    email_text = "\n---\n".join(e.get("mail", "") for e in email_win) or "none"
    return (
        f"USER ID: {user_id}\n"
        f"NAME: {name}\n"
        f"PROFILE: {user_profile.get('description', 'No description available.')}\n\n"
        f"=== USER'S TRANSACTIONS TO ASSESS ===\n{txn_list}\n\n"
        f"=== SMS THREADS ===\n{sms_text}\n\n"
        f"=== EMAIL THREADS ===\n{email_text}\n\n"
        f"=== AUDIO TRANSCRIPTS ===\n{transcript_text}\n\n"
        "Assess comms-derived fraud risk for EACH transaction listed above."
    )


def _run_comms_batch(
    system_prompt: str,
    model_id: str,
    user_id: str,
    user_profile: dict,
    txn_batch: list[dict],
    sms_threads: list[dict],
    email_threads: list[dict],
    transcript_text: str,
) -> str:
    """Run one batch; halve and recurse on context overflow."""
    start, end = _txn_batch_window(
        txn_batch, COMMS_WINDOW_DAYS_BEFORE, COMMS_WINDOW_DAYS_AFTER
    )
    sms_win = filter_sms_by_window(sms_threads, start, end)
    email_win = filter_emails_by_window(email_threads, start, end)
    prompt = _build_comms_prompt(
        user_id, user_profile, txn_batch, sms_win, email_win, transcript_text
    )
    agent = Agent(
        model=make_model(model_id),
        system_prompt=system_prompt,
        callback_handler=None,
        name=f"comms_agent_{user_id}_bs{len(txn_batch)}",
    )
    try:
        return call_agent_with_retry(
            agent, prompt, model_id,
            label=f"comms_agent_{user_id}_bs{len(txn_batch)}",
        )
    except ContextOverflowError:
        if len(txn_batch) <= 1:
            raise
        mid = max(1, len(txn_batch) // 2)
        logger.warning(
            "comms_agent_%s: context overflow, halving %d -> %d",
            user_id, len(txn_batch), mid,
        )
        left = _run_comms_batch(
            system_prompt, model_id, user_id, user_profile,
            txn_batch[:mid], sms_threads, email_threads, transcript_text,
        )
        right = _run_comms_batch(
            system_prompt, model_id, user_id, user_profile,
            txn_batch[mid:], sms_threads, email_threads, transcript_text,
        )
        return f"{left}\n{right}"


def run_comms_agent(
    user_id: str,
    user_profile: dict,
    sms_threads: list[dict],
    email_threads: list[dict],
    audio_transcripts: dict[str, str],
    user_transactions: list[dict],
    legit_domains_sample: str = "",
    model_id: str = MODEL_COMMS_AGENT,
    batch_size: int = BATCH_SIZE_COMMS_AGENT,
) -> str:
    """Analyse communications for one user and return per-transaction comms risk.

    Splits ``user_transactions`` into batches; for each batch, keeps only the
    SMS/emails dated within [earliest_txn - days_before, latest_txn + days_after]
    so prompts stay within the model's context window. Caller is expected to
    have already filtered ``sms_threads``/``email_threads`` to this user.

    Args:
        user_id: The user's sender_id.
        user_profile: Dict with user profile fields from users.json.
        sms_threads: SMS dicts (with key 'sms') addressed to this user.
        email_threads: Email dicts (with key 'mail') addressed to this user.
        audio_transcripts: Dict mapping speaker_name to transcript text.
        user_transactions: Dicts with txn_id, timestamp, amount, txn_type,
            description — the user's transactions to assess.
        legit_domains_sample: Comma-separated sample of known-legit domains.
        model_id: Model to use.
        batch_size: Transactions per LLM call; halved on context overflow.

    Returns:
        Concatenated per-transaction TXN_COMMS_RISK blocks across all batches.
    """
    if not user_transactions:
        return ""
    system_prompt = _COMMS_PROMPT.format(
        legit_domains_sample=legit_domains_sample or "paypal.com, amazon.com, google.com"
    )
    speaker_name = (
        f"{user_profile.get('first_name', '')}_{user_profile.get('last_name', '')}"
        .lower().replace(" ", "_")
    )
    transcript_text = audio_transcripts.get(speaker_name, "none")

    outputs: list[str] = []
    for i in range(0, len(user_transactions), batch_size):
        batch = user_transactions[i: i + batch_size]
        outputs.append(
            _run_comms_batch(
                system_prompt, model_id, user_id, user_profile,
                batch, sms_threads, email_threads, transcript_text,
            )
        )
    return "\n".join(outputs)


# ---------------------------------------------------------------------------
# Identity Agent
# ---------------------------------------------------------------------------

_IDENTITY_PROMPT = """You are an Identity Fraud Analyst operating in Reply Mirror (year 2087).

IMPORTANT TEMPORAL CONTEXT: It is 2087. Birth years in the 2030s–2060s are normal
working-age adults. Do NOT flag these as suspicious. A birth year is only anomalous if
the person would be implausibly young (<18) or implausibly old (>120) in 2087.

You receive a user's verified profile and their transaction history.

Detect synthetic identity or account-takeover signals:
- Name mismatches: transaction descriptions reference a different person
- Address/residence mismatches: transactions reference a different city than registered
- Salary inconsistency: salary payment amounts differ significantly from stated annual salary
- Multiple salary senders (already flagged by signals if present — confirm or escalate)
- Profile-transaction mismatch: spend pattern inconsistent with stated job/income level

Output EXACTLY:
IDENTITY_RISK: low/medium/high
MISMATCH_FLAGS: <comma-separated flags or "none">
SUMMARY: <one sentence>"""


def run_identity_agent(
    user_id: str,
    user_profile: dict,
    user_transactions: list[dict],
    model_id: str = MODEL_COMMS_AGENT,
) -> str:
    """Validate user identity signals against transaction patterns.

    Args:
        user_id: The user's sender_id.
        user_profile: Dict with user profile fields from users.json.
        user_transactions: List of transaction dicts for this user (as CSV rows).
        model_id: Model to use.

    Returns:
        Identity risk assessment text block.
    """
    agent = Agent(
        model=make_model(model_id),
        system_prompt=_IDENTITY_PROMPT,
        callback_handler=None,
        name=f"identity_agent_{user_id}",
    )
    txn_text = "\n".join(
        f"  - {t.get('transaction_id', '?')} | {t.get('timestamp', '?')} | "
        f"\u20ac{float(t.get('amount', 0) or 0):.2f} | {t.get('transaction_type', '?')} | "
        f"{str(t.get('description', ''))[:80]}"
        for t in user_transactions[:30]
    ) or "  (none)"

    residence = user_profile.get("residence", {})
    city = residence.get("city", "unknown") if isinstance(residence, dict) else str(residence)
    prompt = (
        f"USER ID: {user_id}\n"
        f"PROFILE: name={user_profile.get('first_name')} {user_profile.get('last_name')}, "
        f"birth_year={user_profile.get('birth_year')}, "
        f"job={user_profile.get('job')}, "
        f"salary=\u20ac{user_profile.get('salary', 0)}/yr, "
        f"residence={city}, iban={user_profile.get('iban')}\n\n"
        f"TRANSACTIONS:\n{txn_text}\n\n"
        f"Assess identity fraud risk for user {user_id}."
    )
    return call_agent_with_retry(
        agent, prompt, model_id,
        label=f"identity_agent_{user_id}",
    )


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
    return call_agent_with_retry(
        agent,
        f"Analyse these transactions:\n\n{batch_text}",
        model_id,
        label="transaction_agent",
    )
