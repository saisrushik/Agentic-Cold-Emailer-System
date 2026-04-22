import streamlit as st
import os
import sys
import tempfile
from dotenv import load_dotenv

# ── make project root importable ──────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_injection.resume_reader import ResumeReader  # noqa: E402

load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_TRACING"] = "true"

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cold Emailing Agent",
    page_icon="🤖",
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
# Using a sentinel key ensures all state is fresh on every new browser
# session (page refresh), while still surviving widget-triggered reruns.
if "_initialised" not in st.session_state:
    st.session_state._initialised = True
    st.session_state.resume_data = None
    st.session_state.contact_info = None
    st.session_state.llm_provider = "Groq"

# ══════════════════════  SIDEBAR  ══════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # ── LLM Provider selector ─────────────────────────────────────────
    LLM_PROVIDERS = ["OpenAI", "Ollama", "Groq", "Ollama-cloud"]
    selected_provider = st.selectbox(
        "🧠 LLM Provider",
        options=LLM_PROVIDERS,
        index=LLM_PROVIDERS.index(st.session_state.llm_provider),
        help="Choose the LLM backend that will power the cold-email agent.",
    )
    st.session_state.llm_provider = selected_provider
    st.markdown(
        f'<span class="provider-chip">{selected_provider}</span>',
        unsafe_allow_html=True,
    )

    # ── Resume section ────────────────────────────────────────────────
    st.markdown("### 📄 Upload Resume")

    if st.session_state.resume_data is None:
        # No resume loaded — show uploader
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

            with st.spinner("📖 Parsing resume …"):
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
                    st.warning(f"⚠️ Could not extract contact info: {e}")
                st.success("✅ Resume uploaded & parsed!")
                st.rerun()
            else:
                st.error("❌ Failed to parse resume. Please try a different file.")
    else:
        # Resume already loaded — show status + discard
        st.markdown(
            '<div class="success-banner">📄 Resume loaded in session ✔</div>',
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Discard Resume", type="secondary", use_container_width=True):
            st.session_state.resume_data = None
            st.session_state.contact_info = None
            st.rerun()


# ══════════════════════  MAIN AREA  ═══════════════════════════════════
st.markdown('<h1 class="header-title">Agentic Cold Emailing System</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="header-subtitle">'
    "Generate highly personalised cold emails powered by AI"
    "</p>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([2, 1])

# ── Resume Preview ────────────────────────────────────────────────────
with col1:
    st.markdown("#### 📋 Resume Preview")
    if st.session_state.resume_data:
        for i, doc in enumerate(st.session_state.resume_data):
            with st.expander(f"Page {i + 1}", expanded=False):
                st.markdown(doc.page_content)
    else:
        st.info("⬅️ Upload your resume from the sidebar to get started.")

# ── Contact Info ──────────────────────────────────────────────────────
with col2:
    st.markdown("#### 🔗 Extracted Contact Info")
    if st.session_state.contact_info:
        info = st.session_state.contact_info
        contact_rows = {
            "📧 Email": info.get("email"),
            "📞 Phone": info.get("phone"),
            "💼 LinkedIn": info.get("linkedin"),
            "🐙 GitHub": info.get("github"),
            "🌐 Portfolio": info.get("portfolio"),
            "🎓 Scholar": info.get("scholar"),
        }
        with st.expander("📋 View Contact Details", expanded=False):
            for label, value in contact_rows.items():
                if value:
                    st.markdown(f"**{label}:** {value}")
    else:
        st.info("Contact info will appear here after uploading a resume.")

