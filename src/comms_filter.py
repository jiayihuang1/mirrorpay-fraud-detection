"""
Deterministic helpers to trim communications before they reach the LLM.

Filters global SMS/email corpora down to the messages addressed to one user,
strips HTML from email bodies, and supports time-window filtering around a
batch of transactions so prompts stay within the model's context window.
"""

import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Callable

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_STYLE_SCRIPT_RE = re.compile(
    r"<(style|script)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE
)
_DATE_HEADER_RE = re.compile(r"^Date:\s*(.+?)\s*$", re.MULTILINE)
_SMS_TO_RE = re.compile(r"^To:\s*(\S+)\s*$", re.MULTILINE)
_EMAIL_TO_RE = re.compile(r"^To:\s*(.+?)\s*$", re.MULTILINE)


def strip_email_html(raw_mail: str) -> str:
    """Collapse HTML body to plain text while preserving mail headers.

    Args:
        raw_mail: Full mail text including headers and HTML body.

    Returns:
        Original headers plus a compact plain-text body.
    """
    parts = raw_mail.split("\n\n", 1)
    headers = parts[0]
    body = parts[1] if len(parts) > 1 else ""
    body = _STYLE_SCRIPT_RE.sub(" ", body)
    body = _TAG_RE.sub(" ", body)
    body = unescape(body)
    body = _WHITESPACE_RE.sub(" ", body).strip()
    return f"{headers}\n\n{body}" if body else headers


def _parse_sms_date(text: str) -> datetime | None:
    """Parse the Date header of an SMS entry. Returns None on failure."""
    m = _DATE_HEADER_RE.search(text)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1).strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _parse_email_date(text: str) -> datetime | None:
    """Parse an RFC 2822 Date header from an email. Returns None on failure."""
    m = _DATE_HEADER_RE.search(text)
    if not m:
        return None
    try:
        dt = parsedate_to_datetime(m.group(1).strip())
        return dt.replace(tzinfo=None) if dt else None
    except (TypeError, ValueError):
        return None


def _sms_to_phone(text: str) -> str | None:
    m = _SMS_TO_RE.search(text)
    return m.group(1).strip() if m else None


def _email_to_field(text: str) -> str:
    m = _EMAIL_TO_RE.search(text)
    return m.group(1).strip() if m else ""


def build_phone_user_map(
    sms_data: list[dict],
    user_first_names: list[str],
) -> dict[str, str]:
    """Assign each SMS phone number to the user whose first name is most mentioned.

    Args:
        sms_data: List of SMS dicts with key 'sms'.
        user_first_names: First names of all known users.

    Returns:
        Dict mapping phone number to the owning user's first name (lowercased).
    """
    names = {n.lower() for n in user_first_names if n}
    counts: dict[str, Counter] = defaultdict(Counter)
    for entry in sms_data:
        text = entry.get("sms", "")
        phone = _sms_to_phone(text)
        if not phone:
            continue
        lowered = text.lower()
        for name in names:
            if re.search(rf"\b{re.escape(name)}\b", lowered):
                counts[phone][name] += 1
    return {p: c.most_common(1)[0][0] for p, c in counts.items() if c}


def filter_user_comms(
    sms_data: list[dict],
    email_data: list[dict],
    user_first_name: str,
    user_last_name: str,
    phone_user_map: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    """Return only SMS/emails addressed to the given user, with email HTML stripped.

    Args:
        sms_data: All SMS entries.
        email_data: All email entries.
        user_first_name: User's first name.
        user_last_name: User's last name.
        phone_user_map: Output of build_phone_user_map.

    Returns:
        (user_sms, user_emails) — emails already stripped of HTML.
    """
    fn = user_first_name.lower()
    ln = user_last_name.lower()
    phones = {p for p, n in phone_user_map.items() if n == fn}
    user_sms = [
        s for s in sms_data
        if _sms_to_phone(s.get("sms", "")) in phones
    ]
    user_emails: list[dict] = []
    for entry in email_data:
        raw = entry.get("mail", "")
        to_field = _email_to_field(raw).lower()
        if fn in to_field and ln in to_field:
            user_emails.append({"mail": strip_email_html(raw)})
    return user_sms, user_emails


def _filter_by_window(
    comms: list[dict],
    body_key: str,
    parse_date: Callable[[str], datetime | None],
    start_dt: datetime,
    end_dt: datetime,
) -> list[dict]:
    """Keep comms whose parsed date lies inside [start_dt, end_dt].

    Entries with an unparseable date are kept — prefer false-include over drop.
    """
    kept: list[dict] = []
    for entry in comms:
        dt = parse_date(entry.get(body_key, ""))
        if dt is None or start_dt <= dt <= end_dt:
            kept.append(entry)
    return kept


def filter_sms_by_window(
    sms: list[dict], start_dt: datetime, end_dt: datetime
) -> list[dict]:
    """Keep SMS entries whose Date header falls inside the window."""
    return _filter_by_window(sms, "sms", _parse_sms_date, start_dt, end_dt)


def filter_emails_by_window(
    emails: list[dict], start_dt: datetime, end_dt: datetime
) -> list[dict]:
    """Keep email entries whose Date header falls inside the window."""
    return _filter_by_window(emails, "mail", _parse_email_date, start_dt, end_dt)
