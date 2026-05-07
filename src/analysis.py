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
    """Parse numeric and datetime columns.

    Args:
        df: Raw DataFrame from CSV.

    Returns:
        DataFrame with amount, balance_after as float and timestamp as datetime.
    """
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
    """Map sender_id → list of GPS pings (matched via BioTag prefix).

    Args:
        locations: List of location ping dicts.
        df: Transactions DataFrame for sender_id matching.

    Returns:
        Dict mapping sender_id to list of {ts, city} dicts.
    """
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
    """Match a BioTag to a sender_id via exact or prefix match.

    Args:
        biotag: BioTag string from locations.json.
        df: Transactions DataFrame.

    Returns:
        Matched sender_id or None.
    """
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
    if s["shared_iban_count"] >= 2:
        flags.append(f"SHARED_IBAN({s['shared_iban_count']}_senders)")
    if s["circular_transfer"]:
        flags.append("CIRCULAR_TRANSFER")
    if s["location_coherence"] is False:
        flags.append("LOCATION_MISMATCH")

    if s.get("salary_multi_sender_iban"):
        flags.append("SALARY_MULTI_SENDER_IBAN")
    if s.get("salary_duplicate_month"):
        flags.append("SALARY_DUPLICATE_MONTH")
    if s.get("rent_duplicate_month"):
        flags.append("RENT_DUPLICATE_MONTH")
    if s.get("rent_city_mismatch"):
        flags.append("RENT_CITY_MISMATCH")

    flags_str = ", ".join(flags) if flags else "none"
    mean_amt = b.get("mean_amount", 0.0)
    desc = str(s["description"])[:60]
    if s.get("is_salary_txn"):
        desc = f"[SALARY] {desc}"
    elif s.get("is_rent_txn"):
        desc = f"[RENT] {desc}"

    return (
        f"TXN_ID: {txn_id}\n"
        f"  user={user_id} (avg_txn=\u20ac{mean_amt:.0f})\n"
        f"  amount=\u20ac{s['amount']:.2f} (z={s['amount_zscore']:+.1f}), "
        f"type={s['txn_type']}, hour={s['hour']}h\n"
        f"  recipient={s['recipient_id'] or 'N/A'}, "
        f"balance_after=\u20ac{s['balance_after'] or 'N/A'}\n"
        f"  description=\"{desc}\"\n"
        f"  FLAGS: {flags_str}"
    )


def compute_salary_rent_signals(
    df: pd.DataFrame,
    user_residence_map: dict[str, str],
) -> dict[str, dict]:
    """Detect salary and rent anomaly patterns per transaction.

    Salary signals:
    - salary_multi_sender_iban: recipient gets salary from >1 unique sender IBAN (mule signal)
    - salary_duplicate_month: >1 salary payment in the same calendar month

    Rent signals:
    - rent_duplicate_month: >1 rent payment in the same calendar month
    - rent_city_mismatch: rent description city does not match the sender's registered residence

    Args:
        df: Transactions DataFrame (type-coerced).
        user_residence_map: Dict mapping sender_id to lowercase residence city.

    Returns:
        Dict mapping transaction_id to salary/rent signal dict.
    """
    sal_mask = df[COL_DESCRIPTION].str.contains("salary", case=False, na=False)
    rent_mask = df[COL_DESCRIPTION].str.contains("rent", case=False, na=False)

    sal_df = df[sal_mask].copy()
    if not sal_df.empty:
        sal_df["_m"] = sal_df[COL_TIMESTAMP].dt.to_period("M")
        multi_sender = sal_df.groupby(COL_RECIPIENT_ID)[COL_SENDER_IBAN].nunique().gt(1)
        sal_dup_ids: set[str] = set(
            sal_df.groupby([COL_RECIPIENT_ID, "_m"])
            .filter(lambda g: len(g) > 1)[COL_TXN_ID]
            .astype(str)
        )
    else:
        multi_sender = pd.Series(dtype=bool)
        sal_dup_ids = set()

    rent_df = df[rent_mask].copy()
    if not rent_df.empty:
        rent_df["_m"] = rent_df[COL_TIMESTAMP].dt.to_period("M")
        rent_dup_ids: set[str] = set(
            rent_df.groupby([COL_SENDER_ID, "_m"])
            .filter(lambda g: len(g) > 1)[COL_TXN_ID]
            .astype(str)
        )
    else:
        rent_dup_ids = set()

    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        txn_id = str(row[COL_TXN_ID])
        desc = str(row.get(COL_DESCRIPTION, "")).lower()
        sender_id = str(row[COL_SENDER_ID])
        recipient_id = str(row[COL_RECIPIENT_ID])
        is_salary = "salary" in desc
        is_rent = "rent" in desc

        city_mismatch = False
        if is_rent:
            city = user_residence_map.get(sender_id, "").lower()
            city_mismatch = bool(city and city not in desc)

        result[txn_id] = {
            "is_salary_txn": is_salary,
            "is_rent_txn": is_rent,
            "salary_multi_sender_iban": is_salary and bool(multi_sender.get(recipient_id, False)),
            "salary_duplicate_month": txn_id in sal_dup_ids,
            "rent_duplicate_month": txn_id in rent_dup_ids,
            "rent_city_mismatch": city_mismatch,
        }
    return result
