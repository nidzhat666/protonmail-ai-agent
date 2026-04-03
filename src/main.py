import email
import imaplib
import logging
import os
import re
import time
from email.header import decode_header
from enum import Enum

from openai import OpenAI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
IMAP_HOST = os.environ.get("IMAP_HOST", "127.0.0.1")
IMAP_PORT = int(os.environ.get("IMAP_PORT", "1143"))
PROTON_MAIL_USER = os.environ.get("PROTON_MAIL_USER", "")
PROTON_MAIL_PASS = os.environ.get("PROTON_MAIL_PASS", "")
CHECK_INTERVAL_SECONDS = int(os.environ.get("CHECK_INTERVAL_SECONDS", "60"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models for structured OpenAI output
# ---------------------------------------------------------------------------


class EmailCategory(str, Enum):
    WORK = "Work"
    FINANCE = "Finance"
    PERSONAL = "Personal"
    SPAM = "Spam"
    NEWSLETTER = "Newsletter"
    SOCIAL = "Social"
    TRANSACTIONAL = "Transactional"
    OTHER = "Other"


class EmailClassification(BaseModel):
    category: EmailCategory
    summary: str
    priority: int  # 1 (low) – 5 (urgent)
    reasoning: str


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an email classification assistant. Analyze the email and classify it \
into exactly one category. Provide a short summary, a priority score, and your \
reasoning.

Categories:
- Work: professional emails, meetings, project updates, colleague messages
- Finance: banking, invoices, payments, subscriptions, financial statements
- Personal: friends, family, personal matters
- Spam: unsolicited marketing, phishing, scams
- Newsletter: periodic newsletters, digests, blog updates
- Social: social media notifications, community updates
- Transactional: order confirmations, shipping updates, account notifications
- Other: anything that does not fit the above

Priority scale: 1 = low, 2 = below-average, 3 = normal, 4 = high, 5 = urgent.
"""

# ---------------------------------------------------------------------------
# IMAP helpers
# ---------------------------------------------------------------------------


def connect_imap() -> imaplib.IMAP4:
    """Connect to Proton Bridge IMAP and log in."""
    log.info("Connecting to IMAP %s:%s …", IMAP_HOST, IMAP_PORT)
    conn = imaplib.IMAP4(IMAP_HOST, IMAP_PORT)
    conn.login(PROTON_MAIL_USER, PROTON_MAIL_PASS)
    log.info("IMAP login successful.")
    return conn


def ensure_labels(conn: imaplib.IMAP4) -> None:
    """Create Labels/<category> IMAP folders if they don't already exist."""
    status, mailboxes = conn.list()
    existing: set[str] = set()
    if status == "OK" and mailboxes:
        for mb in mailboxes:
            if mb is None:
                continue
            decoded = mb.decode() if isinstance(mb, bytes) else mb
            # Extract the mailbox name (last part after the delimiter)
            match = re.search(r'"([^"]*)"$', decoded)
            if match:
                existing.add(match.group(1))

    for cat in EmailCategory:
        label = f"Labels/{cat.value}"
        if label not in existing:
            log.info("Creating IMAP folder: %s", label)
            conn.create(f'"{label}"')


def _decode_header_value(raw: str | None) -> str:
    if not raw:
        return ""
    parts: list[str] = []
    for fragment, charset in decode_header(raw):
        if isinstance(fragment, bytes):
            parts.append(fragment.decode(charset or "utf-8", errors="replace"))
        else:
            parts.append(fragment)
    return " ".join(parts)


def _extract_body(msg: email.message.Message) -> str:
    """Return the plain-text body of an email, falling back to stripped HTML."""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
        # Fallback: try HTML
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    text = payload.decode(charset, errors="replace")
                    return re.sub(r"<[^>]+>", " ", text)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            if msg.get_content_type() == "text/html":
                return re.sub(r"<[^>]+>", " ", text)
            return text
    return ""


def fetch_unseen_emails(conn: imaplib.IMAP4) -> list[tuple[bytes, dict]]:
    """Fetch UNSEEN emails from INBOX. Returns [(uid, info_dict), …]."""
    conn.select("INBOX")
    status, data = conn.uid("SEARCH", None, "UNSEEN")
    if status != "OK" or not data or not data[0]:
        return []

    uids = data[0].split()
    results: list[tuple[bytes, dict]] = []
    for uid in uids:
        status, msg_data = conn.uid("FETCH", uid, "(RFC822)")
        if status != "OK" or not msg_data or not msg_data[0]:
            continue
        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw)
        info = {
            "subject": _decode_header_value(msg.get("Subject")),
            "from": _decode_header_value(msg.get("From")),
            "date": msg.get("Date", ""),
            "body": _extract_body(msg)[:2000],
        }
        results.append((uid, info))
    return results


def apply_label(conn: imaplib.IMAP4, uid: bytes, category: str) -> None:
    """Copy the email to Labels/<category> to apply the label."""
    label = f'"Labels/{category}"'
    status, _ = conn.uid("COPY", uid, label)
    if status == "OK":
        log.info("Applied label %s to UID %s", category, uid.decode())
    else:
        log.warning("Failed to apply label %s to UID %s", category, uid.decode())


# ---------------------------------------------------------------------------
# OpenAI classification
# ---------------------------------------------------------------------------


def classify_email(
    client: OpenAI, model: str, email_info: dict
) -> EmailClassification | None:
    """Classify a single email using OpenAI structured outputs."""
    user_content = (
        f"From: {email_info['from']}\n"
        f"Subject: {email_info['subject']}\n"
        f"Date: {email_info['date']}\n\n"
        f"{email_info['body']}"
    )
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format=EmailClassification,
        )
        return response.choices[0].message.parsed
    except Exception:
        log.exception("OpenAI classification failed")
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    # Validate required config
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not PROTON_MAIL_USER:
        missing.append("PROTON_MAIL_USER")
    if not PROTON_MAIL_PASS:
        missing.append("PROTON_MAIL_PASS")
    if missing:
        log.error("Missing required environment variables: %s", ", ".join(missing))
        raise SystemExit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)
    conn: imaplib.IMAP4 | None = None

    log.info(
        "Starting Proton-GPT (model=%s, interval=%ss)", OPENAI_MODEL, CHECK_INTERVAL_SECONDS
    )

    while True:
        try:
            # (Re)connect if needed
            if conn is None:
                conn = connect_imap()
                ensure_labels(conn)

            emails = fetch_unseen_emails(conn)
            if not emails:
                log.info("No new emails.")
            else:
                log.info("Found %d new email(s).", len(emails))

            for uid, info in emails:
                log.info(
                    "Processing: [%s] %s", info["from"], info["subject"]
                )
                result = classify_email(client, OPENAI_MODEL, info)
                if result is None:
                    log.warning("Skipping UID %s — classification failed.", uid.decode())
                    continue

                log.info(
                    "Classified → %s (priority %d): %s",
                    result.category.value,
                    result.priority,
                    result.summary,
                )
                apply_label(conn, uid, result.category.value)

        except (imaplib.IMAP4.error, ConnectionError, OSError) as exc:
            log.warning("IMAP connection error: %s — will reconnect.", exc)
            conn = None
        except KeyboardInterrupt:
            log.info("Shutting down.")
            if conn:
                try:
                    conn.logout()
                except Exception:
                    pass
            break
        except Exception:
            log.exception("Unexpected error")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
