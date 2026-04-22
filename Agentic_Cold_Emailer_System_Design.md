# Agentic Cold Emailer System — System Design

## Overview

An agentic cold email system that:
- Reads HR contact data from a CSV file (`email.csv`)
- Ingests an uploaded PDF resume, extracts full text and embedded hyperlinks
- Derives contact info (email, phone, LinkedIn, GitHub, Portfolio, Scholar) from those hyperlinks
- Stores the parsed resume in Streamlit session state for downstream use
- Will build a RAG pipeline for context-aware, personalised email generation
- Supports multiple LLM providers selectable at runtime
- Uses Streamlit for the UI
- Uses LangChain, LangSmith, and LangGraph for orchestration and tracing

---

## Goals

- Automatically generate personalized cold emails to HR contacts
- Use the uploaded resume as the primary knowledge source
- Enable LLM provider selection at runtime (OpenAI, Ollama, Groq, Ollama-cloud)
- Track agent execution with LangSmith
- Build a modular, maintainable system using a proper Python package structure

---

## Current Project Structure

```
Agentic Cold Emailer System/
├── .env                          # API keys (LANGSMITH_API_KEY, provider keys)
├── .python-version               # Python version pin
├── main.py                       # Entrypoint placeholder
├── pyproject.toml                # Project metadata & dependencies
├── requirements.txt              # Pip-installable dependencies
│
├── Data/                         # Sample data files
│   ├── email.csv                 # HR contacts (HR Name, Email, Company, Hiring Role, …)
│   ├── Resume - ML.pdf           # Sample resume (PDF)
│   └── Resume - ML.docx          # Sample resume (DOCX)
│
├── data_injection/               # Python package — data ingestion layer
│   ├── __init__.py               # Re-exports ResumeReader
│   ├── resume_reader.py          # PDF loader + hyperlink extractor (IMPLEMENTED)
│   └── excel_csv.py              # CSV/Excel reader + column extractors (IMPLEMENTED)
│
├── rag_pipeline/                 # Python package — RAG layer (PENDING)
│   └── __init__.py
│
├── emailer_agent/                # Python package — agent + email layer (PENDING)
│   └── __init__.py
│
└── frontend/                     # Python package — Streamlit UI
    ├── __init__.py
    └── app.py                    # Main Streamlit application (IMPLEMENTED)
```

---

## High-Level Architecture

```
┌──────────────┐     ┌──────────────────────┐     ┌───────────────────┐
│   Streamlit  │────▶│   data_injection      │────▶│   rag_pipeline    │
│   frontend   │     │  ResumeReader         │     │  (vector store +  │
│   (app.py)   │     │  excel_csv            │     │   retrieval chain)│
└──────┬───────┘     └──────────────────────┘     └────────┬──────────┘
       │                                                    │
       │ session_state                                      ▼
       │  - resume_data                          ┌───────────────────┐
       │  - contact_info                         │  emailer_agent    │
       │  - llm_provider                         │  (LangGraph DAG)  │
       └────────────────────────────────────────▶└────────┬──────────┘
                                                          │
                                              ┌───────────▼──────────┐
                                              │  LLM Provider        │
                                              │  OpenAI / Groq /     │
                                              │  Ollama / Ollama-    │
                                              │  cloud               │
                                              └──────────────────────┘
```

---

## Implemented Components

### 1. `data_injection/resume_reader.py` — `ResumeReader`

Loads a PDF resume and enriches it with embedded hyperlinks.

| Method | Description |
|---|---|
| `read_resume(file_path)` | Uses `PyMuPDFLoader` (LangChain) to load one `Document` per page. Then uses `PyMuPDF` (`fitz`) directly to extract all hyperlinks per page and appends them to `page_content` and `metadata["hyperlinks"]`. Returns `list[Document]` or `None`. |
| `extract_email(mailto_links)` | Decodes `mailto:` URIs with `urllib.parse.unquote` and extracts email address via regex. |
| `extract_phone(mailto_links)` | Decodes `mailto:` URIs and extracts phone number via regex. |
| `extract_contact_info(resume_data)` | Aggregates hyperlinks from **all pages** (not just page 0). Separates `http` and `mailto:` links, then extracts: email, phone, LinkedIn, GitHub, portfolio (sites.google.com), Google Scholar. Returns a `dict`. |

**Key design decisions:**
- Dual-pass approach: LangChain loader for text, PyMuPDF for hyperlinks
- Hyperlinks collected across ALL pages (not just the first)
- Graceful degradation: missing hyperlinks return empty dict, not an exception

---

### 2. `data_injection/excel_csv.py` — CSV/Excel helpers

Standalone functions to read and extract columns from HR contact spreadsheets.

| Function | Returns |
|---|---|
| `read_csv(file_path)` | `pd.DataFrame` |
| `get_hr_names(df)` | `list` from column `HR Name` |
| `get_emails(df)` | `list` from column `Email` |
| `get_companies(df)` | `list` from column `Company` |
| `get_hiring_roles(df)` | `list` from column `Hiring Role` |
| `get_last_email_sent_dates(df)` | `list` from column `Last Email Sent Date` |
| `get_callback_status(df)` | `list` from column `Received Callback` |

**Expected CSV schema:**

| Column | Description |
|---|---|
| `HR Name` | Recipient name |
| `Email` | HR email address |
| `Company` | Company name |
| `Hiring Role` | Role being hired for |
| `Last Email Sent Date` | Date of last outreach |
| `Received Callback` | Boolean / status |

---

### 3. `frontend/app.py` — Streamlit UI

Entry point: `streamlit run frontend/app.py` (from project root)

#### Session State Keys

| Key | Type | Description |
|---|---|---|
| `_initialised` | `bool` | Sentinel — set on first load of a browser session; ensures state is wiped on page refresh |
| `resume_data` | `list[Document] \| None` | Parsed resume pages from `ResumeReader.read_resume()` |
| `contact_info` | `dict \| None` | Extracted contact details from `ResumeReader.extract_contact_info()` |
| `llm_provider` | `str` | Selected LLM provider (`"Groq"` default) |

#### Sidebar Behaviour

| State | What the user sees |
|---|---|
| No resume loaded | `st.file_uploader` accepting PDF — file is saved to a temp path, parsed, then temp file deleted |
| Resume loaded | Green banner "📄 Resume loaded in session ✔" + **🗑️ Discard Resume** button |
| Discard clicked | `resume_data` and `contact_info` cleared → `st.rerun()` → back to uploader |

> Only one resume can be loaded at a time. The uploader is hidden once a resume is in session state, preventing accidental re-upload.

#### Main Area Layout

```
┌────────────────────────────┬──────────────────────┐
│  📋 Resume Preview          │  🔗 Extracted Contact │
│  (one expander per page,   │  (one expander — View │
│   collapsed by default)    │   Contact Details)    │
└────────────────────────────┴──────────────────────┘
```

- **LLM Provider selector** in sidebar — `st.selectbox` with options: `OpenAI`, `Ollama`, `Groq`, `Ollama-cloud`
- Provider chip displayed below the dropdown for quick visibility
- LangSmith tracing configured via `LANGSMITH_API_KEY` from `.env`

---

## Pending Components

### `rag_pipeline/` — Vector Store & Retrieval

Planned implementation:

- `vector_store.py` — Embed resume `Document` objects into Chroma using the selected provider's embedding model
- `rag_chain.py` — `RetrievalQA` chain that retrieves candidate context per HR contact and generates a personalised email prompt

### `emailer_agent/` — LangGraph Agent

Planned implementation:

- `agent_graph.py` — LangGraph DAG with nodes:
  - `load_contacts` → read HR CSV
  - `embed_profile` → embed resume into Chroma
  - `retrieve_context` → retrieve relevant resume chunks per recipient
  - `generate_email` → call LLM with prompt template
  - `preview_email` → display draft in Streamlit
  - `send_email` *(optional)* → SMTP / transactional API
- `agent_runner.py` — Execute graph with LangSmith tracing per step

---

## Example User Flow (Current)

1. User opens Streamlit UI at `http://localhost:8501`
2. Selects LLM provider from the sidebar dropdown
3. Uploads a PDF resume — parsed and stored in `session_state.resume_data`
4. Contact info (email, phone, LinkedIn, etc.) auto-extracted and shown in "Extracted Contact Info" expander
5. Resume text visible page-by-page in "Resume Preview" expanders

## Example User Flow (Target — End-to-End)

1–4. Same as above
5. System embeds resume into Chroma vector store
6. User uploads `email.csv` with HR contacts
7. For each HR contact, agent retrieves relevant resume context and generates a cold email
8. User previews all drafts in the UI
9. User confirms and sends / exports emails

---

## LangSmith & Tracing

- `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true` set in `.env` and loaded at startup
- Will be used to trace:
  - Prompt templates sent to LLM
  - Retriever queries and returned chunks
  - LangGraph node execution states
  - Model responses and latency

---

## LLM Provider Support

| Provider | Status | Notes |
|---|---|---|
| `Groq` | Default selection | Fast inference, good for drafting |
| `OpenAI` | Selectable | `gpt-4.1-mini`, `gpt-4o-mini` |
| `Ollama` | Selectable | Local model, no API key required |
| `Ollama-cloud` | Selectable | Cloud-hosted Ollama endpoint |

Provider selection is stored in `session_state.llm_provider` and will be used to instantiate the correct LangChain LLM and embedding objects.

---

## Technologies Used

| Library | Purpose |
|---|---|
| `streamlit` | UI — upload, preview, provider selection |
| `langchain-community` | `PyMuPDFLoader` for PDF ingestion |
| `pymupdf` (`fitz`) | Hyperlink extraction from PDF |
| `pandas` | CSV/Excel reading |
| `python-dotenv` | Environment variable loading |
| `langsmith` | Agent tracing and observability |
| `langchain` | Prompt templates, retrieval chains *(planned)* |
| `langgraph` | Agentic DAG workflow *(planned)* |
| `chromadb` | Vector store for resume embeddings *(planned)* |

---

## Next Steps

1. ✅ Build Streamlit UI with LLM provider selection and resume upload
2. ✅ Implement `ResumeReader` with hyperlink extraction
3. ✅ Implement `excel_csv.py` with HR contact column extractors
4. ✅ Refactor into proper Python package structure (`data_injection/`, `rag_pipeline/`, `emailer_agent/`, `frontend/`)
5. ⬜ Implement `rag_pipeline/vector_store.py` — embed resume into Chroma
6. ⬜ Implement `rag_pipeline/rag_chain.py` — retrieval chain for email context
7. ⬜ Implement `emailer_agent/agent_graph.py` — LangGraph DAG
8. ⬜ Add Excel/CSV upload to the Streamlit UI
9. ⬜ Add email preview, approval, and send integration
10. ⬜ Evaluate and iterate on prompt quality using LangSmith traces

---

## Notes

- Keep all API keys in `.env` — never commit them to git
- Session state is fully cleared on browser refresh (sentinel `_initialised` pattern)
- Only one resume can be loaded at a time — discard before re-uploading
- Validate email addresses from CSV before sending
- Keep prompt templates versioned so LangSmith can track prompt changes over time