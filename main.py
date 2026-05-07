"""Main entry point: runs the multi-agent fraud detection pipeline.

Usage:
    python main.py <level>        # run one level (1, 2, or 3)
    python main.py all            # run all available levels in sequence

Output:
    output/level_<N>_output.txt   # fraudulent transaction IDs, one per line
    Prints the Langfuse session ID to stdout for downstream inspection.
"""

import json
import logging
import os
import sys
import uuid
import ulid
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
    compute_salary_rent_signals,
)
from src.comms_filter import build_phone_user_map, filter_user_comms
from src.memory import load_threat_intel, run_memory_agent
from src.orchestrator import run_orchestrator_batch
from src.sub_agents import run_comms_agent
from src.transcription import transcribe_audio_files
from src.utils import (
    ContextOverflowError,
    init_langfuse,
    flush_langfuse,
    validate_output,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _setup_file_logging(level: int) -> None:
    """Attach an INFO-level file handler to mirror all log output to disk.

    Args:
        level: Dataset level number — used to name the log file.
    """
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"level_{level}_run.log"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logger.info("Detailed logs → %s", log_path)


def _load_json(path: Path) -> list:
    """Load a JSON file as a list, returning [] if the file does not exist.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed list, or empty list if file missing.
    """
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _batch(items: list, size: int) -> list[list]:
    """Split a list into fixed-size chunks.

    Args:
        items: List to batch.
        size: Maximum items per batch.

    Returns:
        List of sublists.
    """
    return [items[i: i + size] for i in range(0, len(items), size)]


def run_level(level: int, session_id: str) -> dict[str, int]:
    """Run the full fraud detection pipeline for one dataset level.

    Args:
        level: Dataset level number (1–5).
        session_id: Langfuse session ID for tracing.

    Returns:
        Dict mapping transaction_id to classification (0=legit, 1=fraud).
    """
    _setup_file_logging(level)
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

    # Load users early — needed for salary/rent signals and comms agent
    users_data = _load_json(data_dir / "users.json")
    user_data_map: dict[str, dict] = {}
    user_residence_map: dict[str, str] = {}
    for _u in users_data:
        _iban = _u.get("iban", "")
        _ids = df[df["sender_iban"] == _iban]["sender_id"].unique()
        if len(_ids) > 0:
            _uid = str(_ids[0])
            user_data_map[_uid] = _u
            _res = _u.get("residence", {})
            _city = (_res.get("city", "") if isinstance(_res, dict) else str(_res)).lower()
            user_residence_map[_uid] = _city

    salary_rent = compute_salary_rent_signals(df, user_residence_map)
    for _tid, _extra in salary_rent.items():
        if _tid in signals:
            signals[_tid].update(_extra)

    logger.info(
        "Loaded %d transactions for %d users", len(all_txn_ids), len(baselines)
    )

    # Load legit domains for comms agent URL reference
    legit_domains_path = Path(DATA_DIR) / "legit_transaction_domains.txt"
    legit_domains_sample = ""
    if legit_domains_path.exists():
        _lines = [
            ln.strip() for ln in legit_domains_path.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        legit_domains_sample = ", ".join(_lines[:50])

    # -----------------------------------------------------------------------
    # Phase 0b: Audio transcription (cached to disk)
    # -----------------------------------------------------------------------
    logger.info("Phase 0b: audio transcription...")
    audio_dir = data_dir / "audio"
    audio_transcripts = transcribe_audio_files(audio_dir)
    logger.info("Transcribed %d speakers", len(audio_transcripts))

    # -----------------------------------------------------------------------
    # Phase 1: Communications Agent — one call per user (per-transaction output)
    # -----------------------------------------------------------------------
    logger.info("Phase 1: Communications Agent (%d users)...", len(baselines))
    sms_data = _load_json(data_dir / "sms.json")
    email_data = _load_json(data_dir / "mails.json")

    all_first_names = [u.get("first_name", "") for u in users_data if u.get("first_name")]
    phone_user_map = build_phone_user_map(sms_data, all_first_names)
    logger.info(
        "Pre-filtering comms: %d SMS + %d emails across %d mapped phones",
        len(sms_data), len(email_data), len(phone_user_map),
    )

    user_profiles: dict[str, str] = {}
    for user in users_data:
        user_iban = user.get("iban", "")
        matching_ids = df[df["sender_iban"] == user_iban]["sender_id"].unique()
        user_id = str(matching_ids[0]) if len(matching_ids) > 0 else None

        if not user_id:
            logger.warning(
                "Could not match user %s %s to a sender_id",
                user.get("first_name"),
                user.get("last_name"),
            )
            continue

        # Build this user's transaction list for temporal comms analysis
        _ucols = ["transaction_id", "timestamp", "amount", "transaction_type", "description"]
        _utxns = df[df["sender_id"] == user_id][_ucols].copy()
        user_txns = [
            {
                "txn_id": str(r["transaction_id"]),
                "timestamp": str(r["timestamp"]),
                "amount": float(r["amount"] or 0),
                "txn_type": str(r["transaction_type"]),
                "description": str(r["description"]),
            }
            for _, r in _utxns.iterrows()
        ]

        user_sms, user_emails = filter_user_comms(
            sms_data, email_data,
            user.get("first_name", ""), user.get("last_name", ""),
            phone_user_map,
        )
        logger.info(
            "User %s: filtered %d/%d SMS, %d/%d emails",
            user_id, len(user_sms), len(sms_data),
            len(user_emails), len(email_data),
        )
        profile = run_comms_agent(
            user_id=user_id,
            user_profile=user,
            sms_threads=user_sms,
            email_threads=user_emails,
            audio_transcripts=audio_transcripts,
            user_transactions=user_txns,
            legit_domains_sample=legit_domains_sample,
        )
        user_profiles[user_id] = profile
        logger.info("Comms profile for %s:\n%s", user_id, profile)

    # -----------------------------------------------------------------------
    # Load threat intelligence from previous level (if any)
    # -----------------------------------------------------------------------
    threat_intel = load_threat_intel(level - 1) if level > 1 else None
    if threat_intel:
        logger.info("Loaded threat intelligence from level %d", level - 1)

    # -----------------------------------------------------------------------
    # Phase 2: Dynamic Orchestrator — batched with halve-on-overflow
    # -----------------------------------------------------------------------
    logger.info(
        "Phase 2: Orchestrator classifying %d transactions...", len(all_txn_ids)
    )
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
                user_data=user_data_map,
            )
            all_classifications.update(result)

            # Retry any IDs the model dropped
            _found, missing = validate_output(
                "\n".join(f"{k}: {v}" for k, v in result.items()),
                set(batch),
            )
            if missing:
                logger.warning(
                    "Orchestrator dropped %d IDs — retrying as mini-batch", len(missing)
                )
                mini_result = run_orchestrator_batch(
                    txn_ids=list(missing),
                    signals=signals,
                    baselines=baselines,
                    user_profiles=user_profiles,
                    all_transactions=all_transactions,
                    threat_intel=threat_intel,
                    user_data=user_data_map,
                )
                all_classifications.update(mini_result)

            pending = pending[batch_size:]

        except ContextOverflowError:
            if batch_size <= 1:
                raise RuntimeError("Batch size of 1 still causes context overflow")
            batch_size = max(1, batch_size // 2)
            logger.warning(
                "Context overflow — reducing batch size to %d", batch_size
            )
            # Do not advance pending — retry same batch with smaller size

    logger.info(
        "Classified %d/%d transactions", len(all_classifications), len(all_txn_ids)
    )

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


def _write_output(level: int, classifications: dict[str, int], session_id: str) -> Path:
    """Write fraudulent transaction IDs to the submission file.

    Args:
        level: Dataset level number.
        classifications: Full classification dict.
        session_id: Langfuse session ID — used in the output filename.

    Returns:
        Path to the output file.
    """
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}_level_{level}.txt"

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

    team = os.getenv("TEAM_NAME", "team").replace(" ", "-")
    session_id = f"{team}-{ulid.new().str}"
    logger.info("Langfuse session ID: %s", session_id)
    print(f"\nLangfuse Session ID: {session_id}\n")

    init_langfuse(session_id)

    for level in levels:
        classifications = run_level(level, session_id)
        _write_output(level, classifications, session_id)

    flush_langfuse()
    print(f"\nDone. Session ID for submission: {session_id}")


if __name__ == "__main__":
    main()
