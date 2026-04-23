import streamlit as st
import os
import sys
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
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            with st.spinner("Parsing resume..."):
                reader = ResumeReader()
                resume_data = reader.read_resume(tmp_path)

            os.unlink(tmp_path)

            if resume_data:
                st.session_state.resume_data = resume_data
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
            st.session_state.contact_info = None
            st.session_state.vector_store_db = None
            st.session_state.vector_store_retriever = None
            st.session_state.rag_chain_instance = None
            st.session_state.chat_messages = []
            st.rerun()


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

    def _render_email_results(email_groups, filter_data):
        """Render multiple tailored email previews grouped by company+role."""
        if isinstance(filter_data, dict):
            filters = QueryFilter(**filter_data)
        else:
            filters = filter_data

        # Show applied filters
        filter_parts = []
        if filters.company_types:
            filter_parts.append(f"**Company Type:** {', '.join(filters.company_types)}")
        if filters.hiring_role:
            filter_parts.append(f"**Role:** {filters.hiring_role}")
        if filters.company:
            filter_parts.append(f"**Company:** {filters.company}")

        total_recipients = sum(len(g["contacts"]) for g in email_groups)
        if filter_parts:
            st.info(f"Filters: {' | '.join(filter_parts)} — {total_recipients} recipient(s), {len(email_groups)} email(s)")
        else:
            st.info(f"No specific filters — {len(email_groups)} email(s) for {total_recipients} contact(s)")

        for i, group in enumerate(email_groups):
            email_data = group["email"]
            contacts = group["contacts"]
            if isinstance(email_data, dict):
                email = ColdEmail(**email_data)
            else:
                email = email_data

            company = contacts[0].get("Company", "Unknown") if contacts else "Unknown"
            role = contacts[0].get("Hiring Role", "Unknown") if contacts else "Unknown"
            label = f"{company} — {role} ({len(contacts)} recipient(s))"

            with st.expander(label, expanded=(len(email_groups) == 1)):
                st.markdown(f"**Subject:** {email.subject}")
                st.markdown("---")
                st.markdown(email.greeting)
                st.markdown(email.body)
                st.markdown(email.closing)
                st.markdown(f"*{email.signature}*")
                st.markdown("---")
                st.caption("**Recipients:**")
                for c in contacts:
                    st.caption(
                        f"  {c.get('HR Name/Team', 'N/A')} — "
                        f"{c.get('Email', 'N/A')}"
                    )

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
                    _render_email_results(msg["email_groups"], msg["filters"])
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
                                {"email": email.model_dump(), "contacts": contacts}
                                for email, contacts in email_groups_raw
                            ]
                            thinking_status.update(
                                label="Done", state="complete", expanded=False
                            )
                            response_placeholder.empty()
                            _render_email_results(email_groups, filters)
                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "type": "email",
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

