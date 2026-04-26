"""Gmail SMTP client with PDF attachment support.

Sender credentials are passed in by the caller (the Streamlit UI collects
them from the user — sender email comes from the uploaded resume, app
password is entered in the sidebar). Falls back to environment variables
GMAIL_SENDER / GMAIL_APP_PASSWORD / SENDER_NAME if not supplied, so the
module remains usable from scripts.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr
from typing import Optional


GMAIL_SMTP_HOST = "smtp.gmail.com"
GMAIL_SMTP_PORT = 587


class SMTPConfigError(RuntimeError):
    """Raised when Gmail credentials are missing or invalid."""


def _resolve_credentials(
    sender: Optional[str],
    password: Optional[str],
    sender_name: Optional[str],
) -> tuple[str, str, str]:
    sender = (sender or os.getenv("GMAIL_SENDER", "")).strip()
    password = (password or os.getenv("GMAIL_APP_PASSWORD", "")).strip()
    name = (sender_name or os.getenv("SENDER_NAME", "")).strip() or sender

    if not sender or not password:
        raise SMTPConfigError(
            "Missing Gmail credentials. The sender email is taken from your "
            "uploaded resume, and the Gmail App Password must be entered in "
            "the sidebar (or set via GMAIL_APP_PASSWORD env var)."
        )
    return sender, password, name


def build_message(
    *,
    sender: str,
    sender_name: str,
    to_email: str,
    subject: str,
    body_text: str,
    cc_email: Optional[str] = None,
    attachment_bytes: Optional[bytes] = None,
    attachment_filename: str = "resume.pdf",
) -> EmailMessage:
    """Build an RFC-822 EmailMessage with optional PDF attachment."""
    msg = EmailMessage()
    msg["From"] = formataddr((sender_name, sender))
    msg["To"] = to_email
    if cc_email:
        msg["Cc"] = cc_email
    msg["Subject"] = subject
    msg.set_content(body_text)

    if attachment_bytes:
        msg.add_attachment(
            attachment_bytes,
            maintype="application",
            subtype="pdf",
            filename=attachment_filename,
        )
    return msg


def send_email(
    *,
    to_email: str,
    subject: str,
    body_text: str,
    sender_email: Optional[str] = None,
    sender_password: Optional[str] = None,
    sender_name: Optional[str] = None,
    attachment_bytes: Optional[bytes] = None,
    attachment_filename: str = "resume.pdf",
    cc_sender: bool = True,
) -> dict:
    """Send a single email through Gmail SMTP. Returns a result dict."""
    sender, password, display_name = _resolve_credentials(
        sender_email, sender_password, sender_name
    )
    cc_email = sender if cc_sender else None

    msg = build_message(
        sender=sender,
        sender_name=display_name,
        to_email=to_email,
        subject=subject,
        body_text=body_text,
        cc_email=cc_email,
        attachment_bytes=attachment_bytes,
        attachment_filename=attachment_filename,
    )

    context = ssl.create_default_context()
    with smtplib.SMTP(GMAIL_SMTP_HOST, GMAIL_SMTP_PORT, timeout=30) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(sender, password)
        refused = server.send_message(msg)

    return {
        "from": sender,
        "to": to_email,
        "cc": cc_email,
        "subject": subject,
        "refused_recipients": refused or {},
        "status": "sent",
    }
