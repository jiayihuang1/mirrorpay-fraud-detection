"""
Dynamic Orchestrator — the directing intelligence of the pipeline.

A strands Agent that receives statistical signals and cached user profiles,
then selectively calls @tool functions to gather additional evidence before
making final binary fraud classifications.
"""

import logging
import re

from strands import Agent, tool

from config import MODEL_ORCHESTRATOR, MODEL_TXN_AGENT, BATCH_SIZE_TXN_AGENT
from src.utils import make_model, call_agent_with_retry
from src.analysis import format_txn_summary

logger = logging.getLogger(__name__)

_ORCHESTRATOR_PROMPT = """You are a senior fraud investigator at MirrorPay with 20 years of experience.
You think like a detective, not a rule engine. Your job is to build a coherent narrative
for each transaction — and only flag it as fraud when the evidence tells a convincing story
of deliberate criminal activity, not just statistical noise.

SETTING: It is the year 2087 in Reply Mirror. Citizens born in the 2030s–2060s are normal
working-age adults. Do NOT treat birth years in that range as suspicious.

SIGNAL FLAGS IN TRANSACTION SUMMARIES:
- SALARY_MULTI_SENDER_IBAN: user receives salary from >1 unique sender IBAN (mule indicator)
- SALARY_DUPLICATE_MONTH: >1 salary payment received in same calendar month
- RENT_DUPLICATE_MONTH: >1 rent payment made in same calendar month
- RENT_CITY_MISMATCH: rent payment city in description ≠ user's registered residence

TOOLS — use them when you need more evidence before deciding:
- analyze_financial_patterns(txn_ids, user_id): Detailed pattern analysis vs user's full history
- analyze_recipient_network(recipient_iban): Who else sends to this IBAN? Mule or merchant?
- get_location_check(user_id, txn_id, timestamp, location): GPS coherence at time of transaction
- get_comms_detail(user_id, txn_id): Was the user socially engineered before this transaction?
- analyze_identity(user_id): Name/address/salary consistency check

HOW TO REASON — for each transaction, work through these questions:

1. INNOCENT EXPLANATION FIRST
   What is the most plausible non-fraudulent explanation for this transaction?
   Statistical anomalies are noise unless they form a pattern.
   [NORMAL TRANSACTIONS IN UNUSUAL HOURS - NON-EXHAUSTIVE EXAMPLES]
   There are many perfectly normal reasons for a transaction to occur at an unusual hour. Don't let the clock alone make you suspicious.
   - A salary arriving at 10am is still a salary.
   - A coffee at midnight near a bar is still a coffee.
   - Ridesharing at 3am is unusual but could be a late night out and the driver would be a new recipient.
   
2. EVIDENCE AGAINST THE INNOCENT EXPLANATION
   What specific evidence makes the innocent explanation less credible?
   Weak: a single z-score spike, a mildly unusual hour.
   Strong: new recipient + large amount + GPS contradiction + social engineering all at once.

3. DOES THE EVIDENCE TELL A COHERENT FRAUD STORY?
   Fraud has a shape: there is usually a motive, a method, and a target.
   If the flags don't hang together into a recognisable fraud pattern, they are probably noise.
   Ask: would an experienced colleague agree this looks like a real attack?

4. COMMS AS A MULTIPLIER
   If the comms block shows PRE_TXN_FLAGS=yes with RISK_LEVEL=high — the user was likely
   manipulated. Even a statistically routine transaction becomes suspicious if the user was
   targeted by social engineering immediately before it.
   Comms RISK_LEVEL=low with no flags? Don't let weak statistical signals carry the decision alone.

5. ECONOMIC CALIBRATION
   Scale your certainty requirement to the stakes:
   - Small amounts (<€500): need strong converging evidence to block. False positives on
     routine small transactions erode customer trust for minimal fraud recovery.
   - Large amounts (>€10,000): a moderate but coherent evidence set is sufficient to block.
     The asymmetric cost of missing high-value fraud justifies earlier intervention.
   - BUT: a large salary payment that matches prior salaries is still legitimate regardless
     of amount. Amount alone is never sufficient to classify as fraud.

6. PROFILE-RELATIVE CALIBRATION
   Always compare to THIS user's established pattern, not a generic population.
   A z-score of 3 for a user with highly variable spending is different from the same score
   for a user who makes identical transactions every month.

7. THREAT INTELLIGENCE CARRY-FORWARD
   If prior-level patterns are provided, look for evolved versions of those tactics.
   Fraudsters adapt the surface but rarely change the underlying scheme.
   
FINAL CALL:
- Classify 1 (fraud) only when the evidence forms a coherent, convincing case.
- Classify 0 (legitimate) when the most plausible explanation remains innocent.
- When genuinely uncertain on a high-value transaction after investigation, lean 1.
- When genuinely uncertain on a low-value transaction, lean 0.
- Never let a single isolated signal drive a fraud call. Always ask: what else supports this?

OUTPUT FORMAT — output EXACTLY one line per transaction, ALL transactions:
<full-uuid>: CLASSIFICATION (0 or 1) | CONFIDENCE (high/medium/low) | REASON (one sentence)

No preamble. No summary. Just the classification lines."""


def make_orchestrator_agent(
    signals: dict[str, dict],
    baselines: dict[str, dict],
    user_profiles: dict[str, str],
    all_transactions: list[dict],
    threat_intel: str | None,
    user_data: dict[str, dict] | None = None,
    model_id: str = MODEL_ORCHESTRATOR,
) -> Agent:
    """Build the dynamic orchestrator agent with all tools wired via closures.

    Args:
        signals: Output of compute_statistical_signals().
        baselines: Output of compute_user_baselines().
        user_profiles: Dict mapping user_id to per-transaction comms risk text.
        all_transactions: Raw transaction dicts (list of CSV rows as dicts).
        threat_intel: Threat intelligence brief from previous level, or None.
        user_data: Dict mapping user_id to raw user profile dict from users.json.
        model_id: Orchestrator model to use.

    Returns:
        Configured strands Agent ready to classify a batch of transactions.
    """
    from src.sub_agents import run_transaction_agent, run_identity_agent
    from src.network import get_network_context
    _user_data = user_data or {}

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
        status = "COHERENT" if coherence else "INCOHERENT"
        verb = "supports" if coherence else "contradicts"
        return f"{status}: GPS data {verb} user presence near {location} at {timestamp}."

    @tool
    def get_comms_detail(user_id: str, txn_id: str = "") -> str:
        """Retrieve the per-transaction communications risk for a user.

        Args:
            user_id: The user's sender_id.
            txn_id: Specific transaction UUID to look up (empty = full profile).

        Returns:
            TXN_COMMS_RISK block for the transaction, or full profile if no txn_id.
        """
        profile_text = user_profiles.get(user_id, "")
        if not profile_text:
            return f"No communications data available for user {user_id}."
        if not txn_id:
            return profile_text
        for block in profile_text.split("---"):
            if txn_id.lower() in block.lower():
                return block.strip()
        return f"No specific comms block for {txn_id}. Summary:\n{profile_text[:500]}"

    @tool
    def analyze_identity(user_id: str) -> str:
        """Run identity fraud analysis for a user.

        Validates name, address, salary consistency between profile and transactions.

        Args:
            user_id: The user's sender_id.

        Returns:
            IDENTITY_RISK assessment with mismatch flags.
        """
        user_dict = _user_data.get(user_id, {})
        user_txns = [
            t for t in all_transactions
            if str(t.get("sender_id", "")) == user_id
            or str(t.get("recipient_id", "")) == user_id
        ]
        return run_identity_agent(user_id, user_dict, user_txns)

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
            analyze_identity,
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
    user_data: dict[str, dict] | None = None,
    model_id: str = MODEL_ORCHESTRATOR,
) -> dict[str, int]:
    """Classify one batch of transactions using the dynamic orchestrator.

    Args:
        txn_ids: Transaction UUIDs in this batch.
        signals: Full signals dict (orchestrator only reads relevant entries).
        baselines: Full baselines dict.
        user_profiles: Full user profiles dict (per-transaction comms risk).
        all_transactions: All raw transaction dicts.
        threat_intel: Prior-level threat intelligence brief, or None.
        user_data: Dict mapping user_id to raw user profile dict from users.json.
        model_id: Orchestrator model.

    Returns:
        Dict mapping transaction_id to classification (0 or 1).
    """
    agent = make_orchestrator_agent(
        signals, baselines, user_profiles, all_transactions, threat_intel,
        user_data, model_id,
    )

    batch_lines = []
    for tid in txn_ids:
        if tid in signals:
            batch_lines.append(format_txn_summary(tid, signals, baselines))
            user_id = signals[tid].get("user_id", "")
            # Inline the per-transaction comms block so orchestrator sees it immediately
            comms_block = "unknown"
            profile_text = user_profiles.get(user_id, "")
            if profile_text:
                for block in profile_text.split("---"):
                    if tid.lower() in block.lower():
                        comms_block = block.strip()
                        break
            batch_lines.append(f"  comms_risk:\n{comms_block[:300]}")
    batch_text = "\n\n".join(batch_lines)

    raw = call_agent_with_retry(
        agent,
        f"Classify these {len(txn_ids)} transactions. "
        f"Use tools when you need more evidence.\n\n{batch_text}",
        model_id,
        label="orchestrator_batch",
    )
    logger.info("Orchestrator raw output:\n%s", raw)
    parsed = parse_classifications(raw)
    reasoning = parse_reasoning(raw)
    for tid, cls in sorted(parsed.items()):
        detail = reasoning.get(tid)
        if detail:
            conf, reason = detail
            flag = "FRAUD" if cls == 1 else "legit"
            logger.info("  [%s] %s (%s) — %s", flag, tid, conf, reason)
    logger.info(
        "Orchestrator batch: %d/%d transactions classified",
        len(parsed), len(txn_ids),
    )
    return parsed


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


def parse_reasoning(
    orchestrator_output: str,
) -> dict[str, tuple[str, str]]:
    """Extract confidence and reason for each classified transaction.

    Args:
        orchestrator_output: Raw text output from the orchestrator agent.

    Returns:
        Dict mapping transaction UUID to (confidence, reason) tuple.
    """
    pattern = re.compile(
        r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
        r":\s*[01]\s*\|\s*(high|medium|low)\s*\|\s*(.+)",
        re.IGNORECASE,
    )
    return {
        m.group(1).lower(): (m.group(2).lower(), m.group(3).strip())
        for m in pattern.finditer(orchestrator_output)
    }
