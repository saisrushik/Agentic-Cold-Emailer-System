import streamlit as st
import os
import sys
import time
import uuid
import tempfile
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler

# ── make project root importable ──────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag_pipeline.rag_chain import RagChain, EmailGenerator, ColdEmail, QueryFilter
from rag_pipeline.vector_store import VectorStore
from data_injection.resume_reader import ResumeReader
from data_injection.excel_csv import ExcelCsvReader
from emailer_agent.agent import send_one as agent_send_one

load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_TRACING"] = "true"

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cold Emailing Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a premium look ─────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stFileUploader label {
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* Header gradient text */
    .header-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        line-height: 1.2;
        display: block;
        margin-bottom: 0.2rem;
    }
    .header-subtitle {
        color: #888;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.2rem;
    }

    /* Card wrapper */
    .info-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
    }

    /* Success banner */
    .success-banner {
        background: linear-gradient(135deg, #00c897 0%, #00b4d8 100%);
        color: #fff;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1rem;
    }

    /* Provider chip */
    .provider-chip {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: #fff !important;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    /* Discard button */
    div[data-testid="stButton"] button[kind="secondary"] {
        background: transparent;
        border: 1px solid #e05252;
        color: #e05252 !important;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s ease;
    }
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background: #e05252;
        color: #fff !important;
    }

    /* Reduce default Streamlit top padding */
    .block-container {
        padding-top: 3rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session-state: initialise once per browser session ───────────────
if "_initialised" not in st.session_state:
    st.session_state._initialised = True
    st.session_state.resume_data = None
    st.session_state.resume_bytes = None
    st.session_state.resume_filename = "resume.pdf"
    st.session_state.contact_info = None
    st.session_state.llm_provider = "Groq"
    st.session_state.model_params = {
        "model": None,
        "provider": None,
        "temperature": None,
        "max_tokens": None,
        "max_retries": None,
    }
    st.session_state.hr_records = None
    st.session_state.vector_store_db = None
    st.session_state.vector_store_retriever = None
    st.session_state.rag_chain_instance = None
    st.session_state.chat_messages = []
    st.session_state.session_id = "default_session"
    st.session_state.gmail_app_password = ""
    st.session_state.sender_name_override = ""


# ══════════════════════  SIDEBAR  ══════════════════════════════════════
with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    # ── LLM Provider ──────────────────────────────────────────────────
    LLM_PROVIDERS = ["OpenAI", "Ollama", "Groq", "Ollama-cloud"]
    selected_provider = st.selectbox(
        "LLM Provider",
        options=LLM_PROVIDERS,
        index=LLM_PROVIDERS.index(st.session_state.llm_provider),
        help="Choose the LLM backend that will power the cold-email agent.",
    )
    st.session_state.llm_provider = selected_provider


    st.markdown(
        f'<span class="provider-chip">{selected_provider}</span>',
        unsafe_allow_html=True,
    )

    if selected_provider == "OpenAI":
        api_key=st.text_input("Enter your OpenAI API key:",type="password")

    # ── Model parameters ──────────────────────────────────────────────
    st.markdown("### Model Parameters")
    
    model_options={
            "openai": ["gpt-4o","gpt-4.1"],
            "groq":["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "qwen/qwen3-32b"],
            "ollama": ["gemma4:latest", "gemma4:31b"],
            "ollama-cloud":["gemma4:31b-cloud", "deepseek-v3.2:cloud", "gpt-oss:120b-cloud"]
        }
    
    provider_key = st.session_state.llm_provider.lower()
    if provider_key in model_options:
        st.session_state.model_params["model"] = st.selectbox(
            f"Select {st.session_state.llm_provider} Model",
            options=model_options[provider_key],
            index=0,
        )
    
    st.session_state.model_params["temperature"]=st.slider("Temperature",min_value=0.0,max_value=1.0,value=0.3,step=0.1)
    
    st.session_state.model_params["max_tokens"]=st.number_input("Max Tokens",min_value=128,max_value=4096,value=1024,step=128)
    
    st.session_state.model_params["max_retries"]=st.number_input("Max Retries",min_value=0,max_value=5,value=2,step=1)

    # Always sync the provider into model_params
    st.session_state.model_params["provider"] = st.session_state.llm_provider

    # ── Resume upload ───────────────────────────────────────────────
    st.markdown("### Upload Resume")

    if st.session_state.resume_data is None:
        uploaded_file = st.file_uploader(
            "Drop your PDF resume here",
            type=["pdf"],
            help="Upload your resume in PDF format. "
                 "Hyperlinks and contact info will be extracted automatically.",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            with st.spinner("Parsing resume..."):
                reader = ResumeReader()
                resume_data = reader.read_resume(tmp_path)

            os.unlink(tmp_path)

            if resume_data:
                st.session_state.resume_data = resume_data
                st.session_state.resume_bytes = file_bytes
                st.session_state.resume_filename = uploaded_file.name or "resume.pdf"
                try:
                    contact = reader.extract_contact_info(resume_data)
                    st.session_state.contact_info = contact if contact else None
                except Exception as e:
                    st.session_state.contact_info = None
                    st.warning(f"Could not extract contact info: {e}")
                st.success("Resume uploaded and parsed.")
                st.rerun()
            else:
                st.error("Failed to parse resume. Please try a different file.")
    else:
        st.markdown(
            '<div class="success-banner">Resume loaded in session</div>',
            unsafe_allow_html=True,
        )
        if st.button("Discard Resume", type="secondary", use_container_width=True):
            st.session_state.resume_data = None
            st.session_state.resume_bytes = None
            st.session_state.contact_info = None
            st.session_state.vector_store_db = None
            st.session_state.vector_store_retriever = None
            st.session_state.rag_chain_instance = None
            st.session_state.chat_messages = []
            st.rerun()

    # ── Sender credentials (Gmail SMTP) ──────────────────────────────
    st.markdown("### Sender (Gmail)")
    sender_email_from_resume = (
        (st.session_state.contact_info or {}).get("email") if st.session_state.contact_info else None
    )
    if sender_email_from_resume:
        st.markdown(
            f"**From:** `{sender_email_from_resume}`  \n"
            f"<small>Auto-detected from your resume</small>",
            unsafe_allow_html=True,
        )
    else:
        st.caption(
            "Upload a resume that contains your email so it can be used "
            "as the sender address."
        )

    st.session_state.sender_name_override = st.text_input(
        "Sender display name",
        value=st.session_state.get("sender_name_override", ""),
        placeholder="e.g. Sai Srigiri",
        help="Shown in the From header. Defaults to your email if left blank.",
    )

    st.session_state.gmail_app_password = st.text_input(
        "Gmail App Password",
        value=st.session_state.get("gmail_app_password", ""),
        type="password",
        help=(
            "Required to send mail through Gmail SMTP from your address. "
            "Generate one at https://myaccount.google.com/apppasswords "
            "(2-Step Verification must be enabled). Stored only in this "
            "browser session — never saved to disk."
        ),
    )

    if sender_email_from_resume and st.session_state.gmail_app_password:
        st.markdown(
            '<div class="success-banner">Sender ready</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════  MAIN AREA  ═══════════════════════════════════
st.markdown('<h1 class="header-title">Agentic Cold Emailing System</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="header-subtitle">'
    "Generate highly personalised cold emails powered by AI"
    "</p>",
    unsafe_allow_html=True,
)

# ── Load HR data from Excel/CSV ──────────────────────────────────────
CSV_PATH = os.path.join(PROJECT_ROOT, "Data", "email.csv")

if st.session_state.hr_records is None and os.path.exists(CSV_PATH):
    excel_reader = ExcelCsvReader(CSV_PATH)
    excel_reader.read_csv()
    st.session_state.hr_records = excel_reader.get_all_records()

# ── Create Vector Store DB Embeddings ────────────────────────────────
if st.session_state.vector_store_db is None and st.session_state.resume_data is not None:
    with st.spinner("Creating embeddings in Pinecone Vector DB..."):
        try:
            vector_instance = VectorStore(st.session_state.resume_data)
            vector_store_db = vector_instance.create_vector_store()
            if vector_store_db is not None:
                st.session_state.vector_store_db = vector_store_db
                st.session_state.vector_store_retriever = vector_store_db.as_retriever()
                st.success("Vector store created successfully.")
            else:
                st.error("Failed to create vector store. Check Pinecone API key.")
        except Exception as e:
            st.error(f"Vector store creation failed: {e}")

# ── Build RAG Chain instance (once retriever is ready) ───────────────
if (
    st.session_state.vector_store_retriever is not None
    and st.session_state.model_params.get("model") is not None
    and st.session_state.rag_chain_instance is None
):
    st.session_state.rag_chain_instance = RagChain(
        model_params=st.session_state.model_params,
        vector_store_retriever=st.session_state.vector_store_retriever,
    )


# ── Custom thinking callback: shows LLM / retriever activity live ────
class StreamlitThinkingHandler(BaseCallbackHandler):
    """Writes chain / retriever / LLM activity into an st.status widget
    and streams LLM tokens into a live expandable view."""

    def __init__(self, status):
        self.status = status
        self._current_tokens: list[str] = []
        self._step = 0

    def _log(self, title: str, detail: str = ""):
        self._step += 1
        self.status.update(label=f"Thinking — Step {self._step}: {title}", state="running")
        self.status.markdown(f"**Step {self._step}: {title}**")
        if detail:
            self.status.markdown(detail)

    # Retriever events
    def on_retriever_start(self, serialized, query, **kwargs):
        self._log("Retrieving resume context", f"Query: `{query}`")

    def on_retriever_end(self, documents, **kwargs):
        n = len(documents) if documents else 0
        self.status.markdown(f"→ Retrieved {n} resume chunk(s)")

    # LLM events
    def on_llm_start(self, serialized, prompts, **kwargs):
        name = (serialized or {}).get("name") or (serialized or {}).get("id", ["LLM"])[-1]
        preview = prompts[0] if prompts else ""
        preview = preview[:400] + ("..." if len(preview) > 400 else "")
        self._log(f"Calling {name}", f"```\n{preview}\n```")
        self._current_tokens = []
        self._token_box = self.status.empty()

    def on_llm_new_token(self, token: str, **kwargs):
        self._current_tokens.append(token)
        # live stream the tokens
        joined = "".join(self._current_tokens)
        self._token_box.markdown(f"**Streaming response:**\n\n{joined}")

    def on_llm_end(self, response, **kwargs):
        if self._current_tokens:
            self.status.markdown("→ LLM response complete")
        self._current_tokens = []

    def on_llm_error(self, error, **kwargs):
        self.status.markdown(f"**LLM error:** {error}")

    # Chain events
    def on_chain_start(self, serialized, inputs, **kwargs):
        name = (serialized or {}).get("name") or (serialized or {}).get("id", ["Chain"])[-1]
        if name and name not in ("RunnableSequence", "RunnableParallel", "RunnableLambda"):
            self._log(f"Running chain: {name}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name", "tool")
        self._log(f"Using tool: {name}", f"Input: `{input_str}`")


col1, col2 = st.columns([2, 1])

# ── Left Column: Resume Preview + Unified Chat ───────────────────────
with col1:
    st.markdown("#### Resume Preview")
    if st.session_state.resume_data:
        for i, doc in enumerate(st.session_state.resume_data):
            with st.expander(f"Page {i + 1}", expanded=False):
                st.markdown(doc.page_content)
    else:
        st.info("Upload your resume from the sidebar to get started.")

    # ── Unified Chat (RAG + Email Generation) ─────────────────────────

    def _send_one_email(email_idx_key: str, group: dict) -> None:
        """Invoke the LangGraph emailer for a single contact."""
        email_data = group["email"]
        contact = group["contacts"][0]
        sender_email = ((st.session_state.contact_info or {}).get("email")) or None
        sender_password = (st.session_state.get("gmail_app_password") or "").strip() or None
        sender_name = (st.session_state.get("sender_name_override") or "").strip() or None

        if not sender_email:
            group["status"] = "failed"
            group["error"] = (
                "No sender email detected in the uploaded resume. "
                "Re-upload a resume that includes your email."
            )
            return
        if not sender_password:
            group["status"] = "failed"
            group["error"] = (
                "Gmail App Password is missing. Enter it in the sidebar "
                "(Sender section) before sending."
            )
            return

        try:
            result = agent_send_one(
                to_email=contact.get("Email", ""),
                recipient_name=contact.get("HR Name/Team", "") or "",
                company=contact.get("Company", "") or "",
                role=contact.get("Hiring Role", "") or "",
                subject=email_data["subject"],
                greeting=email_data["greeting"],
                body=email_data["body"],
                closing=email_data["closing"],
                signature=email_data["signature"],
                resume_bytes=st.session_state.get("resume_bytes"),
                attachment_filename=st.session_state.get("resume_filename", "resume.pdf"),
                cc_sender=True,
                sender_email=sender_email,
                sender_password=sender_password,
                sender_name=sender_name,
                thread_id=email_idx_key,
            )
            if result.get("status") == "sent":
                group["status"] = "sent"
                group["error"] = None
            else:
                group["status"] = "failed"
                group["error"] = result.get("error", "Unknown error")
        except Exception as e:  # noqa: BLE001
            group["status"] = "failed"
            group["error"] = str(e)

    def _render_email_results(msg_id: str, email_groups: list, filter_data) -> None:
        """Render editable email previews with Approve / Reject / Send controls."""
        if isinstance(filter_data, dict):
            filters = QueryFilter(**filter_data)
        else:
            filters = filter_data

        filter_parts = []
        if filters.company_types:
            filter_parts.append(f"**Company Type:** {', '.join(filters.company_types)}")
        if filters.hiring_role:
            filter_parts.append(f"**Role:** {filters.hiring_role}")
        if filters.company:
            filter_parts.append(f"**Company:** {filters.company}")

        total_recipients = sum(len(g["contacts"]) for g in email_groups)
        if filter_parts:
            st.info(
                f"Filters: {' | '.join(filter_parts)} — "
                f"{total_recipients} recipient(s), {len(email_groups)} email(s)"
            )
        else:
            st.info(
                f"No specific filters — {len(email_groups)} email(s) for "
                f"{total_recipients} contact(s)"
            )

        if not st.session_state.get("resume_bytes"):
            st.warning(
                "Resume PDF not available in session — re-upload it from the sidebar "
                "before sending so it can be attached."
            )

        sender_email_ready = ((st.session_state.contact_info or {}).get("email")) or None
        if not sender_email_ready:
            st.warning(
                "Sender email not detected in your resume. Re-upload a resume "
                "containing your email so it can be used as the From address."
            )
        if not (st.session_state.get("gmail_app_password") or "").strip():
            st.warning(
                "Gmail App Password is required to send. Enter it in the sidebar "
                "under **Sender (Gmail)**."
            )

        # ── Bulk send button ──
        pending_count = sum(1 for g in email_groups if g.get("status", "pending") == "pending")
        approved_count = sum(1 for g in email_groups if g.get("status") == "approved")

        bulk_col1, bulk_col2 = st.columns([3, 1])
        with bulk_col2:
            send_all_clicked = st.button(
                f"Send All Approved ({approved_count})",
                key=f"send_all_{msg_id}",
                disabled=approved_count == 0,
                use_container_width=True,
            )
        with bulk_col1:
            st.caption(
                f"{approved_count} approved · {pending_count} pending · "
                f"{sum(1 for g in email_groups if g.get('status') == 'sent')} sent · "
                f"{sum(1 for g in email_groups if g.get('status') == 'rejected')} rejected · "
                f"{sum(1 for g in email_groups if g.get('status') == 'failed')} failed"
            )

        if send_all_clicked:
            progress = st.progress(0.0, text="Sending approved emails...")
            approved_groups = [g for g in email_groups if g.get("status") == "approved"]
            total = len(approved_groups)
            for i, group in enumerate(approved_groups):
                contact = group["contacts"][0]
                progress.progress(
                    i / max(total, 1),
                    text=f"Sending to {contact.get('Email','?')} ({i + 1}/{total})",
                )
                if i > 0:
                    time.sleep(2.0)  # 2s delay between sends per Gmail rate limits
                _send_one_email(f"{msg_id}-{i}", group)
            progress.progress(1.0, text="Done.")
            st.rerun()

        # ── Per-email cards ──
        for i, group in enumerate(email_groups):
            email_data = group["email"]
            contacts = group["contacts"]
            contact = contacts[0]
            company = contact.get("Company", "Unknown")
            role = contact.get("Hiring Role", "Unknown")
            hr_name = contact.get("HR Name/Team", "Unknown")
            hr_email = contact.get("Email", "N/A")

            status = group.get("status", "pending")
            badge = {
                "pending": "🟡 pending",
                "approved": "🟢 approved",
                "sent": "✅ sent",
                "rejected": "⛔ rejected",
                "failed": "❌ failed",
            }.get(status, status)

            label = f"{badge}  ·  {company} — {role}  ·  {hr_name} <{hr_email}>"
            expand_default = (status in ("pending", "failed")) and len(email_groups) <= 5

            key_prefix = f"{msg_id}-{i}"
            editing_key = f"editing_{key_prefix}"
            if editing_key not in st.session_state:
                st.session_state[editing_key] = False

            with st.expander(label, expanded=expand_default):
                if status in ("sent", "rejected"):
                    st.markdown(f"**Subject:** {email_data['subject']}")
                    st.markdown("---")
                    st.markdown(email_data["greeting"])
                    st.markdown(email_data["body"])
                    st.markdown(email_data["closing"])
                    st.markdown(f"*{email_data['signature']}*")
                    if status == "sent" and group.get("error"):
                        st.warning(group["error"])
                    continue

                if status == "failed" and group.get("error"):
                    st.error(group["error"])

                if st.session_state[editing_key]:
                    new_subject = st.text_input(
                        "Subject", value=email_data["subject"], key=f"sub_{key_prefix}"
                    )
                    new_greeting = st.text_input(
                        "Greeting", value=email_data["greeting"], key=f"gr_{key_prefix}"
                    )
                    new_body = st.text_area(
                        "Body", value=email_data["body"], height=220, key=f"body_{key_prefix}"
                    )
                    new_closing = st.text_input(
                        "Closing", value=email_data["closing"], key=f"cl_{key_prefix}"
                    )
                    new_signature = st.text_area(
                        "Signature",
                        value=email_data["signature"],
                        height=80,
                        key=f"sig_{key_prefix}",
                    )
                    save_col, cancel_col = st.columns(2)
                    if save_col.button(
                        "Save Changes", key=f"save_{key_prefix}", use_container_width=True
                    ):
                        email_data["subject"] = new_subject
                        email_data["greeting"] = new_greeting
                        email_data["body"] = new_body
                        email_data["closing"] = new_closing
                        email_data["signature"] = new_signature
                        st.session_state[editing_key] = False
                        st.rerun()
                    if cancel_col.button(
                        "Cancel", key=f"cancel_{key_prefix}", use_container_width=True
                    ):
                        st.session_state[editing_key] = False
                        st.rerun()
                else:
                    st.markdown(f"**Subject:** {email_data['subject']}")
                    st.markdown("---")
                    st.markdown(email_data["greeting"])
                    st.markdown(email_data["body"])
                    st.markdown(email_data["closing"])
                    st.markdown(f"*{email_data['signature']}*")
                    st.markdown("---")
                    sender_email_display = (
                        ((st.session_state.contact_info or {}).get("email")) or "(not detected)"
                    )
                    st.caption(
                        f"**From:** {sender_email_display}  ·  "
                        f"**To:** {hr_name} <{hr_email}>  ·  "
                        f"**CC:** {sender_email_display} (auto)  ·  "
                        f"**Attachment:** {st.session_state.get('resume_filename', 'resume.pdf')}"
                    )

                    btn_edit, btn_approve, btn_send, btn_reject = st.columns(4)
                    if btn_edit.button(
                        "Edit", key=f"edit_{key_prefix}", use_container_width=True
                    ):
                        st.session_state[editing_key] = True
                        st.rerun()
                    if btn_approve.button(
                        "Approve",
                        key=f"approve_{key_prefix}",
                        use_container_width=True,
                        disabled=status == "approved",
                    ):
                        group["status"] = "approved"
                        st.rerun()
                    if btn_send.button(
                        "Approve & Send",
                        key=f"send_{key_prefix}",
                        use_container_width=True,
                        type="primary",
                    ):
                        group["status"] = "approved"
                        with st.spinner(f"Sending to {hr_email}..."):
                            _send_one_email(key_prefix, group)
                        st.rerun()
                    if btn_reject.button(
                        "Reject", key=f"rej_{key_prefix}", use_container_width=True
                    ):
                        group["status"] = "rejected"
                        st.rerun()

    st.markdown("#### Chat")

    rag_ready = (
        st.session_state.rag_chain_instance is not None
        and st.session_state.model_params.get("model") is not None
    )

    # Container for chat history — always rendered ABOVE the input
    messages_container = st.container()

    chat_input = st.chat_input(
        "Ask about your resume or request cold emails...",
        disabled=not rag_ready,
    )

    if not rag_ready:
        st.caption("Upload a resume and wait for embeddings to enable the chat.")

    # Render existing history into the top container
    with messages_container:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                if msg.get("type") == "email":
                    _render_email_results(
                        msg["msg_id"], msg["email_groups"], msg["filters"]
                    )
                else:
                    st.markdown(msg.get("content", ""))

    if chat_input and rag_ready:
        # Append + render the user message into the top container
        st.session_state.chat_messages.append({"role": "user", "content": chat_input})
        with messages_container:
            with st.chat_message("user"):
                st.markdown(chat_input)

            # Assistant block — thinking + streamed tokens live above the input
            with st.chat_message("assistant"):
                thinking_status = st.status("Thinking...", expanded=True)
                st_callback = StreamlitThinkingHandler(thinking_status)
                response_placeholder = st.empty()

                try:
                    email_gen = EmailGenerator(
                        model_params=st.session_state.model_params,
                        vector_store_retriever=st.session_state.vector_store_retriever,
                    )

                    # Classify intent and extract filters
                    with thinking_status:
                        st.write("Classifying your request...")
                    query_filter = email_gen.parse_query(
                        chat_input, callbacks=[st_callback]
                    )

                    if query_filter.is_email_request:
                        if st.session_state.hr_records:
                            with thinking_status:
                                st.write("Filtering HR contacts and drafting emails...")
                            email_groups_raw, filters = email_gen.generate_emails(
                                user_query=chat_input,
                                hr_records=st.session_state.hr_records,
                                filters=query_filter,
                                callbacks=[st_callback],
                            )
                            email_groups = [
                                {
                                    "email": email.model_dump(),
                                    "contacts": contacts,
                                    "status": "pending",
                                    "error": None,
                                }
                                for email, contacts in email_groups_raw
                            ]
                            thinking_status.update(
                                label="Done", state="complete", expanded=False
                            )
                            response_placeholder.empty()
                            msg_id = uuid.uuid4().hex[:8]
                            _render_email_results(msg_id, email_groups, filters)
                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "type": "email",
                                "msg_id": msg_id,
                                "email_groups": email_groups,
                                "filters": filters.model_dump(),
                            })
                        else:
                            msg = "HR contact data not found. Place email.csv in the Data/ folder."
                            thinking_status.update(
                                label="No HR data", state="error", expanded=False
                            )
                            response_placeholder.warning(msg)
                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "type": "text",
                                "content": msg,
                            })
                    else:
                        # General RAG question — stream tokens so the user sees progress
                        with thinking_status:
                            st.write("Retrieving resume context and generating answer...")

                        streamed_text = ""
                        try:
                            for chunk in st.session_state.rag_chain_instance.stream(
                                user_input=chat_input,
                                session_id=st.session_state.session_id,
                                callbacks=[st_callback],
                            ):
                                if chunk:
                                    streamed_text += chunk
                                    response_placeholder.markdown(streamed_text + "▌")
                            response_placeholder.markdown(streamed_text)
                        except Exception:
                            # Fallback to non-streaming invoke if streaming fails
                            streamed_text = st.session_state.rag_chain_instance.invoke(
                                user_input=chat_input,
                                session_id=st.session_state.session_id,
                                callbacks=[st_callback],
                            )
                            response_placeholder.markdown(streamed_text)

                        thinking_status.update(
                            label="Done", state="complete", expanded=False
                        )
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "type": "text",
                            "content": streamed_text,
                        })
                except Exception as e:
                    error_msg = f"Error: {e}"
                    thinking_status.update(
                        label="Error", state="error", expanded=True
                    )
                    response_placeholder.error(error_msg)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "type": "text",
                        "content": error_msg,
                    })


# ── Right Column: Contact Info ────────────────────────────────────────
with col2:
    st.markdown("#### Extracted Contact Info")
    if st.session_state.contact_info:
        info = st.session_state.contact_info
        contact_rows = {
            "Email": info.get("email"),
            "Phone": info.get("phone"),
            "LinkedIn": info.get("linkedin"),
            "GitHub": info.get("github"),
            "Portfolio": info.get("portfolio"),
            "Scholar": info.get("scholar"),
        }
        with st.expander("View Contact Details", expanded=False):
            for label, value in contact_rows.items():
                if value:
                    st.markdown(f"**{label}:** {value}")
    else:
        st.info("Contact info will appear here after uploading a resume.")

