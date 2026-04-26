"""LangGraph emailer agent.

Pipeline (traced end-to-end via LangSmith when LANGSMITH_TRACING=true):

    validate  →  attach_resume  →  send_email  →  record_result

Human-in-the-loop approval is handled in the Streamlit UI: the graph is only
invoked AFTER the user clicks "Approve & Send" for a given email. This keeps
state simple on Streamlit Cloud (no background workers, no checkpoint store).
"""

from __future__ import annotations

import re
import time
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from emailer_agent.smtp_client import send_email, SMTPConfigError


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class EmailerState(TypedDict, total=False):
    # Inputs
    to_email: str
    recipient_name: str
    company: str
    role: str
    subject: str
    greeting: str
    body: str
    closing: str
    signature: str
    resume_bytes: Optional[bytes]
    attachment_filename: str
    cc_sender: bool
    sender_email: Optional[str]
    sender_password: Optional[str]
    sender_name: Optional[str]

    # Computed
    composed_body: str
    send_result: dict
    error: str
    status: str  # "validated" | "attached" | "sent" | "failed"


# ── Nodes ─────────────────────────────────────────────────────────────


def _node_validate(state: EmailerState) -> EmailerState:
    to_email = (state.get("to_email") or "").strip()
    if not to_email or not EMAIL_RE.match(to_email):
        return {"status": "failed", "error": f"Invalid recipient email: {to_email!r}"}
    if not state.get("subject", "").strip():
        return {"status": "failed", "error": "Subject is empty."}
    if not state.get("body", "").strip():
        return {"status": "failed", "error": "Body is empty."}
    return {"status": "validated"}


def _node_attach(state: EmailerState) -> EmailerState:
    """Compose the plain-text email body and confirm the resume is attached."""
    parts = [
        state.get("greeting", "").strip(),
        "",
        state.get("body", "").strip(),
        "",
        state.get("closing", "").strip(),
        "",
        state.get("signature", "").strip(),
    ]
    composed = "\n".join(p for p in parts if p is not None)
    has_resume = bool(state.get("resume_bytes"))
    return {
        "composed_body": composed,
        "status": "attached" if has_resume else "validated",
    }


def _node_send(state: EmailerState) -> EmailerState:
    try:
        result = send_email(
            to_email=state["to_email"],
            subject=state["subject"],
            body_text=state["composed_body"],
            sender_email=state.get("sender_email"),
            sender_password=state.get("sender_password"),
            sender_name=state.get("sender_name"),
            attachment_bytes=state.get("resume_bytes"),
            attachment_filename=state.get("attachment_filename", "resume.pdf"),
            cc_sender=state.get("cc_sender", True),
        )
        return {"send_result": result, "status": "sent"}
    except SMTPConfigError as e:
        return {"status": "failed", "error": str(e)}
    except Exception as e:  # noqa: BLE001 — surface any SMTP error in UI
        return {"status": "failed", "error": f"SMTP error: {e}"}


def _node_record(state: EmailerState) -> EmailerState:
    # Hook for future logging / persistence. Currently a no-op terminator.
    return {}


def _route_after_validate(state: EmailerState) -> str:
    return END if state.get("status") == "failed" else "attach_resume"


def _route_after_send(state: EmailerState) -> str:
    return "record_result"


# ── Graph builder ─────────────────────────────────────────────────────


def build_emailer_graph():
    g = StateGraph(EmailerState)
    g.add_node("validate", _node_validate)
    g.add_node("attach_resume", _node_attach)
    g.add_node("send_email", _node_send)
    g.add_node("record_result", _node_record)

    g.add_edge(START, "validate")
    g.add_conditional_edges(
        "validate", _route_after_validate, {END: END, "attach_resume": "attach_resume"}
    )
    g.add_edge("attach_resume", "send_email")
    g.add_edge("send_email", "record_result")
    g.add_edge("record_result", END)
    return g.compile()


# Module-level compiled graph (re-used across calls; LangSmith picks it up)
EMAILER_GRAPH = build_emailer_graph()


# ── Public API ────────────────────────────────────────────────────────


def send_one(
    *,
    to_email: str,
    recipient_name: str,
    company: str,
    role: str,
    subject: str,
    greeting: str,
    body: str,
    closing: str,
    signature: str,
    resume_bytes: Optional[bytes],
    attachment_filename: str = "resume.pdf",
    cc_sender: bool = True,
    sender_email: Optional[str] = None,
    sender_password: Optional[str] = None,
    sender_name: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> EmailerState:
    """Run the LangGraph emailer for a single contact (HITL-approved upstream)."""
    config = {"configurable": {"thread_id": thread_id or f"{company}-{role}-{to_email}"}}
    return EMAILER_GRAPH.invoke(
        {
            "to_email": to_email,
            "recipient_name": recipient_name,
            "company": company,
            "role": role,
            "subject": subject,
            "greeting": greeting,
            "body": body,
            "closing": closing,
            "signature": signature,
            "resume_bytes": resume_bytes,
            "attachment_filename": attachment_filename,
            "cc_sender": cc_sender,
            "sender_email": sender_email,
            "sender_password": sender_password,
            "sender_name": sender_name,
        },
        config=config,
    )


def send_many_with_delay(emails: list[dict], delay_seconds: float = 2.0):
    """Yield (input, result) for each email, sleeping `delay_seconds` between sends."""
    for i, payload in enumerate(emails):
        if i > 0:
            time.sleep(delay_seconds)
        result = send_one(**payload)
        yield payload, result
