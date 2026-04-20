# 🤖 Agentic Cold Emailer System

> An AI-powered, agentic cold email generation system that combines **RAG (Retrieval-Augmented Generation)**, **LangGraph**, **LangChain**, and **ChromaDB** to automatically generate personalized cold emails to HR contacts — driven by your resume and profile.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Data Schema](#data-schema)
- [Component Details](#component-details)
- [Installation & Setup](#installation--setup)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [Workflow](#workflow)
- [LLM Provider Support](#llm-provider-support)
- [Roadmap & Future Enhancements](#roadmap--future-enhancements)
- [Contributing](#contributing)

---

## Overview

The **Agentic Cold Emailer System** is a fully agentic AI pipeline designed to automate and personalize the cold outreach process for job seekers. Instead of manually crafting emails for each HR contact, this system:

1. **Reads your HR contact list** from a CSV/Excel file (HR Name, Email, Company, Hiring Role, Last Email Sent Date).
2. **Ingests your resume** (PDF or DOCX) and embeds it into a vector store (ChromaDB).
3. **Runs an agentic RAG pipeline** — for each HR contact, it retrieves the most relevant parts of your resume/profile and generates a deeply personalized cold email.
4. **Presents a Streamlit UI** for uploading files, selecting LLM providers, previewing generated drafts, and optionally sending emails.
5. **Traces every step** via LangSmith for full observability of prompts, retrievals, and outputs.

This system is ideal for candidates actively applying for AI/ML, data science, or software engineering roles who want to scale their personalized outreach.

---

## Key Features

| Feature | Description |
|---|---|
| 🗂️ **HR Contact Ingestion** | Read HR data from CSV/Excel with columns: `HR Name`, `Email`, `Company`, `Hiring Role`, `Last Email Sent Date` |
| 📄 **Resume Parsing** | Ingest PDF or DOCX resume, extract text, and chunk it for vector embedding |
| 🧠 **RAG Pipeline** | ChromaDB-backed retrieval to fetch candidate profile context for each email |
| 🤖 **Agentic Workflow** | LangGraph-powered multi-step agent: validate → embed → retrieve → generate → preview |
| 🎛️ **Multi-LLM Support** | Switch between OpenAI, Groq, Google Gemini, HuggingFace, or local models at runtime |
| 🖥️ **Streamlit UI** | User-friendly interface to upload files, select providers, preview emails, and track status |
| 🔍 **LangSmith Tracing** | Full execution tracing: prompts, retrievals, model responses, and agent steps |
| 🔒 **Secure Config** | All credentials managed via `.env` file, never hardcoded |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                        │
│  [Upload CSV] [Upload Resume] [Select LLM] [Generate] [Preview]     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │       Agent Orchestrator     │
              │    (LangGraph - agent_graph) │
              └──┬──────────┬──────────┬────┘
                 │          │          │
    ┌────────────▼──┐  ┌────▼─────┐  ┌▼──────────────┐
    │ Data Ingestion │  │  RAG     │  │ Email Generator│
    │                │  │ Pipeline │  │                │
    │ excel_reader   │  │          │  │  LLM Provider  │
    │ resume_reader  │  │ ChromaDB │  │  (OpenAI/Groq/ │
    └────────────────┘  │ Retriever│  │  Gemini/HF)    │
                        └──────────┘  └────────────────┘
                             │
                    ┌────────▼────────┐
                    │   LangSmith     │
                    │   (Tracing &    │
                    │  Observability) │
                    └─────────────────┘
```

### Agent Node Flow (LangGraph)

```
[load_contacts] → [embed_profile] → [retrieve_context] → [generate_email] → [preview_email] → [send_email]
```

---

## Project Structure

```
Agentic Cold Emailer System/
│
├── 📁 Data/                            # Input data directory (gitignored)
│   ├── email.csv                       # HR contact list (HR Name, Email, Company, Hiring Role, Last Email Sent Date)
│   ├── Resume - ML.pdf                 # Candidate resume (PDF format)
│   └── Resume - ML.docx               # Candidate resume (DOCX format)
│
├── 📁 Data Injection/                  # Data ingestion modules
│   ├── excel_reader.py                 # Reads HR contact CSV/Excel; extracts names, emails, companies, roles
│   ├── resume_reader.py                # Parses resume PDF/DOCX into text chunks (WIP)
│   └── excel_reder.ipynb               # Jupyter notebook for testing excel reader
│
├── 📁 RAG/                             # RAG pipeline (WIP)
│   └── (vector_store.py, rag_chain.py) # ChromaDB ingestion & retrieval chain
│
├── 📁 Emailer Agent/                   # LangGraph agent workflow (WIP)
│   └── (agent_graph.py, agent_runner.py, provider_config.py)
│
├── 📁 frontend/                        # Streamlit UI
│   └── app.py                          # Main Streamlit application (WIP)
│
├── main.py                             # Application entry point
├── pyproject.toml                      # Project metadata and dependencies (managed by uv)
├── requirements.txt                    # pip-compatible dependency list
├── .env                                # Environment variables (API keys — gitignored)
├── .gitignore                          # Git exclusions (.env, Data/, .venv)
├── .python-version                     # Python version pin (3.12+)
├── uv.lock                             # Locked dependency tree (uv)
└── Agentic_Cold_Emailer_System_Design.md  # Detailed system design document
```

---

## Tech Stack

### Core AI / LLM

| Library | Role |
|---|---|
| `langchain` | Orchestration, prompt templates, chains |
| `langchain-core` | Base abstractions for LLMs, retrievers |
| `langchain-community` | Community integrations |
| `langchain-openai` | OpenAI & Azure OpenAI LLM/Embeddings |
| `langchain-groq` | Groq inference (LLaMA, Mixtral, Gemma) |
| `langchain-huggingface` | HuggingFace Hub models |
| `langchain-text-splitters` | Document chunking for vector ingestion |
| `google-generativeai` | Google Gemini models |
| `chromadb` | Local vector store for resume embeddings |

### Agentic Workflow

| Library | Role |
|---|---|
| `langgraph` | Multi-step agentic graph (nodes + edges) |
| `langsmith` | Tracing, prompt logging, observability |

### Data & UI

| Library | Role |
|---|---|
| `pandas` | DataFrame operations for CSV/Excel reading |
| `openpyxl` | Excel file engine for `pandas.read_excel` |
| `numpy` | Numerical support |
| `pypdf` | PDF resume parsing |
| `python-dotenv` | Loading `.env` credentials |
| `pydantic` | Data validation and schema models |
| `streamlit` | Web UI for file uploads and email previews |

### Dev Tooling

| Tool | Role |
|---|---|
| `uv` | Fast Python package manager & virtual env |
| `ipykernel` | Jupyter notebook support for local testing |

---

## Data Schema

### HR Contact File (`Data/email.csv`)

The primary input data file must have the following columns:

| Column | Type | Description | Example |
|---|---|---|---|
| `HR Name` | `str` | Full name of the HR contact | `Alice Johnson` |
| `Email` | `str` | HR contact email address | `alice@company.com` |
| `Company` | `str` | Company or organization name | `TechCorp Inc.` |
| `Hiring Role` | `str` | Target job role the HR is hiring for | `AI Engineer` |
| `Last Email Sent Date` | `date` | Date last cold email was sent (optional) | `2026-04-01` |

**Sample:**
```csv
HR Name,Email,Company,Hiring Role,Last Email Sent Date
Alice Johnson,alice@techcorp.com,TechCorp Inc.,AI Engineer,2026-04-01
Bob Smith,bob@dataco.com,DataCo,Python Developer,
```

### Resume Files (`Data/`)

- Accepted formats: **PDF** (`.pdf`) and **DOCX** (`.docx`)
- The resume is parsed, chunked, embedded, and stored in ChromaDB
- Profile sections used for retrieval: Education, Experience, Skills, Certifications, Career Goals

---

## Component Details

### `Data Injection/excel_reader.py`

Reads the HR contact CSV file and provides helper functions to extract each field:

```python
read_excel(file_path)           # Returns pandas DataFrame from CSV/Excel
get_hr_names(df)                # Extracts 'HR Name' column
get_emails(df)                  # Extracts 'Email' column
get_companies(df)               # Extracts 'Company' column
get_hiring_roles(df)            # Extracts 'Hiring Role' column
get_last_email_sent_dates(df)   # Extracts 'Last Email Sent Date' column
```

### `Data Injection/resume_reader.py` *(WIP)*

Will parse the uploaded resume (PDF or DOCX) and extract structured text sections for embedding.

### `RAG/vector_store.py` *(Planned)*

- Initializes ChromaDB collection
- Embeds resume chunks using the selected provider's embedding model
- Stores metadata (section type, page number) for filtered retrieval

### `RAG/rag_chain.py` *(Planned)*

Builds the LangChain retrieval chain:
- Uses `ChromaDB` retriever to fetch relevant resume context
- Constructs a prompt template that merges:
  - HR contact name, company, and role
  - Retrieved candidate profile snippets
  - Target job value proposition
  - Call-to-action and contact details

### `Emailer Agent/agent_graph.py` *(Planned)*

Defines the LangGraph agentic workflow with the following nodes:

| Node | Action |
|---|---|
| `load_contacts` | Load HR contacts from CSV |
| `embed_profile` | Parse and embed resume into ChromaDB |
| `retrieve_context` | Retrieve relevant candidate context for each contact |
| `generate_email` | Generate personalized cold email via LLM |
| `preview_email` | Display draft for user review |
| `send_email` | Optionally send via SMTP or email API |

### `frontend/app.py` *(Planned)*

Streamlit UI with:
- **File uploaders**: CSV (HR contacts) + Resume (PDF/DOCX)
- **Text inputs**: Optional profile summary, target job details
- **Provider select box**: Choose LLM (OpenAI, Groq, Gemini, HuggingFace)
- **Generate button**: Triggers the LangGraph agent pipeline
- **Preview panel**: Review all generated email drafts
- **Send/Export controls**: Approve and dispatch emails

---

## Installation & Setup

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or `pip`

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "Agentic Cold Emailer System"
```

### 2. Install Dependencies

**Using `uv` (recommended):**
```bash
uv sync
```

**Using `pip`:**
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the `.env` template and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials (see [Environment Variables](#environment-variables)).

### 4. Prepare Input Data

Place the following files in the `Data/` directory:

```
Data/
├── email.csv          # Your HR contact list
├── Resume - ML.pdf    # Your resume (PDF)
└── Resume - ML.docx   # Your resume (DOCX) — optional
```

---

## Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# === OpenAI ===
OPENAI_API_KEY=your_openai_api_key

# === Groq ===
GROQ_API_KEY=your_groq_api_key

# === Google Gemini ===
GOOGLE_API_KEY=your_google_api_key

# === HuggingFace ===
HUGGINGFACEHUB_ACCESS_TOKEN=your_hf_token

# === LangSmith (Tracing) ===
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=agentic-cold-emailer

# === Email Sending (optional) ===
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

> **⚠️ Warning:** Never commit your `.env` file to version control. It is already listed in `.gitignore`.

---

## Running the Application

### Run the Streamlit UI

```bash
streamlit run frontend/app.py
```

### Run the Entry Point (CLI)

```bash
python main.py
```

### Run with `uv`

```bash
uv run streamlit run frontend/app.py
```

### Test Data Injection Modules

```bash
# Test Excel reader
python "Data Injection/excel_reader.py"

# Or use the Jupyter notebook
jupyter notebook "Data Injection/excel_reder.ipynb"
```

---

## Workflow

Here is the end-to-end usage flow once the system is fully built:

```
1. Open the Streamlit UI (streamlit run frontend/app.py)
        │
        ▼
2. Upload your HR contact CSV (with HR Name, Email, Company, Hiring Role)
        │
        ▼
3. Upload your Resume (PDF or DOCX)
        │
        ▼
4. Select your LLM provider (OpenAI / Groq / Gemini / HuggingFace)
        │
        ▼
5. Optionally enter a custom profile summary and target job description
        │
        ▼
6. Click "Generate Emails"
        │
        ├─ System embeds resume into ChromaDB
        ├─ For each HR contact:
        │       ├─ Retrieves relevant resume context
        │       └─ Generates a personalized cold email via LLM
        │
        ▼
7. Preview all generated email drafts in the UI
        │
        ▼
8. Approve, edit, export as .txt/.csv, or send directly via SMTP
```

---

## LLM Provider Support

The system is designed to support multiple LLM providers, switchable at runtime from the UI:

| Provider | Models | Use Case |
|---|---|---|
| **OpenAI** | `gpt-4.1-mini`, `gpt-4o-mini` | High-quality email generation |
| **Groq** | `llama-3`, `mixtral-8x7b`, `gemma` | Fast, low-latency inference |
| **Google Gemini** | `gemini-pro`, `gemini-1.5` | Google's multimodal models |
| **HuggingFace** | Any Hub model | Open-source model experimentation |
| **Local** | Ollama / LM Studio | Offline / private inference |

Provider selection is handled via a factory pattern in `provider_config.py`, keeping the LLM and Embedding class selection decoupled from the rest of the system.

---

## Roadmap & Future Enhancements

- [ ] **Resume Parser** — Complete `resume_reader.py` to parse PDF/DOCX sections
- [ ] **ChromaDB Integration** — Implement `vector_store.py` for resume embedding & retrieval
- [ ] **RAG Chain** — Build `rag_chain.py` with prompt templates and retrieval
- [ ] **LangGraph Agent** — Define full agentic graph in `agent_graph.py`
- [ ] **Streamlit UI** — Build complete `app.py` with all upload and preview controls
- [ ] **LangSmith Tracing** — Add `langsmith_tracker.py` for full execution tracing
- [ ] **Email Sending** — SMTP / SendGrid / Gmail API integration with rate limiting
- [ ] **Personalization Layer** — Use recipient company mission and role pain points
- [ ] **Content Safety Filter** — Detect and remove spammy language before sending
- [ ] **Follow-up Scheduler** — Track replies and auto-schedule follow-up sequences
- [ ] **Cost Management** — Track token usage per provider; cache common embeddings
- [ ] **Dashboard Metrics** — Emails generated, sent, open-rate, response-rate tracking

---

## Contributing

Contributions, suggestions, and bug reports are welcome!

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## License

This project is for educational and personal use. Please ensure you comply with the terms of service of any LLM provider APIs you use.

---

<p align="center">
  Built with ❤️ using LangChain · LangGraph · ChromaDB · Streamlit
</p>
