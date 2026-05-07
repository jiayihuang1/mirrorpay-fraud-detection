"""
Memory Agent — synthesises fraud patterns after each level into a
threat intelligence brief loaded by the orchestrator at the next level.
"""

import logging
from pathlib import Path

from strands import Agent

from config import MODEL_MEMORY_AGENT, MEMORY_DIR
from src.utils import make_model, call_agent_with_retry

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
                if v is True or (
                    isinstance(v, (int, float))
                    and k.endswith("_count")
                    and v > 2
                )
            ]
            fraud_summaries.append(
                f"TXN {tid[:8]}: user={user_id}, amount=\u20ac{sig.get('amount', 0):.0f}, "
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
        brief = call_agent_with_retry(
            agent, prompt, model_id,
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
