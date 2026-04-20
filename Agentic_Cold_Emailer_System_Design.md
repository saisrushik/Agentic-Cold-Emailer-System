# Agentic Cold Emailer System Design

## Overview
This document describes a design for an agentic cold email system that:
- reads HR contact data from Excel
- ingests an uploaded resume
- builds a profile knowledge base in Chroma
- creates a RAG pipeline for context-aware email generation
- supports multiple LLM providers
- uses Streamlit for UI
- uses Langchain, LangSmith, and LangGraph for orchestration and tracing

---

## Goals
- Automatically generate personalized cold emails to HR / HR teams
- Use uploaded resume and profile details as the knowledge source
- Enable provider selection for LLM inference
- Track agent execution and data flow with LangSmith
- Build a modular, maintainable system with streamlit UI

---

## High-Level Architecture

1. **Data Ingestion**
   - Read HR contact information from Excel.
   - Read resume upload and optionally parse it into sections.
   - Store profile details in a vector store (Chroma) after embedding.

2. **Vector Storage**
   - Use Chroma to persist resume/profile embeddings.
   - Index resume text and optionally candidate metadata.

3. **RAG Pipeline**
   - Use Langchain to build a retrieval chain.
   - Use a retriever over Chroma to fetch relevant profile context for each email.
   - Construct prompts that merge HR contact details + candidate profile.

4. **Agentic Workflow**
   - Use LangGraph to define an agentic graph of actions.
   - Agents can perform steps: validate email data, retrieve context, generate a draft, and optionally send or preview.
   - Use LangSmith for tracing executions, prompts, responses, and observability.

5. **UI with Streamlit**
   - Upload Excel file containing HR email addresses and company details.
   - Upload resume file.
   - Enter profile details, job target, and email preferences.
   - Select LLM provider and model.
   - Review and approve generated emails.
   - Track logs, status, and action history.

6. **LLM Provider Support**
   - Support providers such as OpenAI, Azure OpenAI, Anthropic, Hugging Face, or local models.
   - Allow selecting provider at runtime from Streamlit dropdown.
   - Keep provider configuration modular.

---

## Component Breakdown

### 1. Data Ingestion
- `excel_reader.py`
  - Read Excel using `pandas.read_excel`.
  - Validate columns like `name`, `email`, `company`, `role`, `notes`.
- `resume_parser.py`
  - Accept uploaded resume file.
  - Optionally parse text from PDF or DOCX.
  - Normalize profile sections.

### 2. Embedding & Vector Store
- `vector_store.py`
  - Use Langchain `Chroma` vector store.
  - Embed resume/profile text using the selected LLM provider's embedding model.
  - Store metadata for retrieval.
- Candidate profile is represented as:
  - education
  - experience
  - skills
  - certifications
  - career goals

### 3. RAG Pipeline
- `rag_chain.py`
  - Build a `RetrievalQA` or `RetrievalQAWithSources` chain.
  - Use retriever from Chroma.
  - Add prompt templates for personalized email generation.
- Prompt structure:
  - HR recipient name/company details
  - candidate summary from resume
  - target role and value proposition
  - call-to-action and contact details

### 4. Agent-Orchestration
- `agent_graph.py`
  - Define nodes for tasks: `load_contacts`, `embed_profile`, `retrieve_context`, `generate_email`, `preview_email`, `send_email`.
  - Connect steps with LangGraph.
- `agent_runner.py`
  - Execute the graph.
  - Use LangSmith tracing for each agent step.

### 5. Streamlit UI
- `app.py`
  - UI components:
    - file uploader for Excel
    - file uploader for resume
    - text input for profile details
    - select box for LLM provider and model
    - button for `Generate Emails`
    - preview panel for generated drafts
    - send/queue controls
  - Show execution trace and results.

### 6. Provider Selection
- `provider_config.py`
  - Maintain provider configurations in a central place.
  - Example providers:
    - `openai`: `gpt-4.1-mini`, `gpt-4o-mini`
    - `azure_openai`: `gpt-4.1-mini`
    - `anthropic`: `claude-3-mini`
    - `local`: any compatible local LLM
- Use an adapter or factory to select the `LLM` and `Embedding` classes dynamically.

---

## Suggested File Structure

- `app.py`
- `excel_reader.py`
- `resume_parser.py`
- `vector_store.py`
- `rag_chain.py`
- `agent_graph.py`
- `provider_config.py`
- `langsmith_tracker.py`
- `requirements.txt`
- `Agentic_Cold_Emailer_System_Design.md`

---

## Example Flow

1. User opens Streamlit UI.
2. Uploads resume and Excel with HR contacts.
3. Chooses an LLM provider.
4. Enters optional profile summary and target job details.
5. System embeds resume/profile into Chroma.
6. System reads each contact from Excel.
7. For each recipient, the agent retrieves profile context and generates a personalized email.
8. User previews all drafts.
9. User confirms and optionally exports or sends emails.

---

## LangSmith & Tracing
- Use LangSmith to track:
  - prompt templates
  - model responses
  - retriever queries
  - execution states for LangGraph nodes
- Benefits:
  - debug prompt quality
  - inspect retrieval relevance
  - measure provider performance

---

## Enhancements & Improvements

### 1. Email Sending Integration
- Add support for SMTP or transactional email APIs (SendGrid, Mailgun, Gmail API).
- Include sending throttling and rate-limiting.
- Add send-status reporting, bounce handling, and retry loops.

### 2. Personalization & Safety
- Add a personalization layer that uses recipient company mission, role, and pain points.
- Use a content filter / safety step before sending.
- Add disallowed phrase detection to prevent spammy wording.

### 3. Profile Enrichment
- Enrich the profile using structured data from LinkedIn or parsed resume sections.
- Add skill keyword extraction, summary generation, and achievements highlighting.
- Store enriched metadata in Chroma for better retrieval.

### 4. Conversation Tracking
- Capture replies and follow-up history if using an email send integration.
- Add a follow-up scheduler and pipeline for sequential outreach.

### 5. Provider & Cost Management
- Allow cost-aware provider selection, e.g. low-cost preview mode and higher-quality send mode.
- Track token usage and prompt length.
- Cache embeddings and common prompts to reduce repeated cost.

### 6. Monitoring & Metrics
- Add dashboard metrics: emails generated, emails sent, success rate, average response time.
- Track LangSmith traces for quality drift and prompt performance.

---

## Recommended Technologies
- Langchain: orchestration, prompts, retrieval
- LangGraph: agentic step definition and workflow control
- LangSmith: tracing, prompt logging, execution metadata
- Chroma: vector store for resume/profile data
- Streamlit: UI for upload, preview, and control
- Pandas / OpenPyXL: Excel ingestion
- PyPDF2 / python-docx: optional resume parsing
- SMTP / transactional email API: email delivery

---

## Next Steps
1. Define the exact Excel columns and resume schema.
2. Build the Streamlit upload and provider-selection UI.
3. Implement Chroma ingestion and LLM provider abstraction.
4. Create the LangGraph workflow and connect it to LangSmith.
5. Add email preview, export, and send integration.
6. Evaluate quality and improve prompts with a small test set.

---

## Notes
- Keep provider credentials secure and configurable via environment variables.
- Use separate Chroma collections if you want multiple candidate profiles.
- Validate email addresses before sending.
- Keep prompt templates versioned so LangSmith can trace prompt changes.
 