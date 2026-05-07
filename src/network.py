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
        return (
            f"RECIPIENT_IBAN: {recipient_iban}\n"
            "NETWORK_CLASS: unknown\nCONFIDENCE: low\n"
            "EVIDENCE: No transactions found.\nRISK_SIGNAL: 0.0"
        )

    senders = {str(t.get("sender_id", "")) for t in related}
    total_amount = sum(float(t.get("amount", 0) or 0) for t in related)
    descriptions = list(
        {str(t.get("description", "")) for t in related if t.get("description")}
    )[:5]

    summary = (
        f"Recipient IBAN: {recipient_iban}\n"
        f"Total transactions: {len(related)}\n"
        f"Distinct senders: {len(senders)} — {', '.join(sorted(senders)[:10])}\n"
        f"Total amount received: \u20ac{total_amount:.2f}\n"
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
