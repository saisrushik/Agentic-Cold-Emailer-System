from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from pydantic import BaseModel, Field
from typing import Optional
from itertools import groupby
from data_injection.excel_csv import ExcelCsvReader

import os
from dotenv import load_dotenv


load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_TRACING"] = "true"


# ── Pydantic Structured Output Models ─────────────────────────────────

class ColdEmail(BaseModel):
    subject: str = Field(description="Email subject line — concise and attention-grabbing")
    greeting: str = Field(
        description=(
            "Greeting line. If the recipient is a named person use 'Dear <name>,' "
            "(e.g., 'Dear Priya,'). If the recipient is a team or the name looks like a "
            "team/department, use 'Dear Hiring Team,' or 'Dear Talent Acquisition Team,'."
        )
    )
    body: str = Field(
        description=(
            "Main email body — professional, non-spammy, tailored to the company type, "
            "role, and candidate resume. Must NOT contain any URLs, hyperlinks, or web addresses."
        )
    )
    closing: str = Field(description="Professional closing statement with a call to action")
    signature: str = Field(
        description=(
            "Email signature with candidate name and contact details (phone/email only). "
            "Must NOT contain any URLs, hyperlinks, or web addresses."
        )
    )


class QueryFilter(BaseModel):
    is_email_request: bool = Field(
        default=False,
        description=(
            "True if the user wants to generate, draft, write, compose, or send a cold email. "
            "False if the user is asking a general question about their resume or anything else."
        ),
    )
    company_types: Optional[list[str]] = Field(
        default=None,
        description=(
            "List of company type filters extracted from the query. "
            "Examples: ['Startup'], ['Product Based', 'Startup'], ['FinTech'], ['GCCs']. "
            "Recognized types: Startup, Product Based, IT Services, IT Services & Consultancy, "
            "FinTech, GCCs, Unicorn, HealthTech. "
            "If MULTIPLE types are mentioned return ALL as a list. None if not specified."
        ),
    )
    hiring_role: Optional[str] = Field(
        default=None,
        description="Hiring role filter (e.g., 'AI Engineer', 'Python Developer', 'ML Engineer'). None if not specified.",
    )
    company: Optional[str] = Field(
        default=None,
        description="Specific company name filter (e.g., 'Google', 'ABC'). Do NOT confuse company-type words like 'startup' with company names. None if not specified.",
    )


# ── LLM Builder ───────────────────────────────────────────────────────

def build_llm(model_params: dict, streaming: bool = False, callbacks: list | None = None):
    model = model_params.get("model")
    temperature = model_params.get("temperature", 0.3)
    max_tokens = model_params.get("max_tokens", 1024)
    max_retries = model_params.get("max_retries", 2)
    provider = model_params.get("provider", "groq")

    kwargs = {}
    if callbacks:
        kwargs["callbacks"] = callbacks
    if streaming:
        kwargs["streaming"] = True

    if provider.lower() == "groq":
        return ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            **kwargs,
        )
    elif provider.lower() == "openai":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            **kwargs,
        )
    elif provider.lower() in ("ollama", "ollama-cloud"):
        return ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


# ── History-Aware Conversational RAG Chain ────────────────────────────

class RagChain:
    def __init__(self, model_params: dict, vector_store_retriever):
        self.model_params = model_params
        self.vector_store_retriever = vector_store_retriever
        self.store: dict[str, ChatMessageHistory] = {}

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def get_conversational_rag_chain(self, streaming: bool = False, callbacks: list | None = None):
        llm = build_llm(self.model_params, streaming=streaming, callbacks=callbacks)

        contextualize_system_prompt = (
            "Given the chat history and the latest user question, "
            "which might reference prior context in the chat history, "
            "formulate a standalone question that can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm, self.vector_store_retriever, contextualize_prompt
        )

        qa_system_prompt = (
            "You are an expert cold-email copywriter for job seekers. "
            "Your task is to generate professional, concise, and personalized cold emails "
            "that a candidate can send to HR professionals or Talent Acquisition teams.\n\n"
            "Guidelines:\n"
            "- Each email must be tailored to the specific HR contact, company, company type, and hiring role.\n"
            "- The tone should be professional, confident, and non-spammy.\n"
            "- Keep emails concise (150-250 words for the body).\n"
            "- Highlight the candidate's relevant skills and experience from the retrieved resume context below.\n"
            "- Include a clear call-to-action (e.g., requesting a brief call or meeting).\n"
            "- Do NOT use generic templates — each email should feel unique and genuine.\n"
            "- Do NOT use any links in the email like linkedin, github, portfolio, or any website links.\n"
            "- Adapt the language based on the company type (e.g., more formal for enterprise, "
            "slightly casual for startups).\n\n"
            "Retrieved Resume Context:\n"
            "---\n"
            "{context}\n"
            "---\n\n"
            "Use the resume context above to craft emails. "
            "If the user provides HR contact details, generate one email per contact. "
            "Answer any follow-up questions using the resume context and chat history."
            "At the end of the email, add a copyright notice '@Agentic-Cold-Emailer-System developed by Sai Srushik' to indicate it was AI-generated."
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        return conversational_rag_chain

    def invoke(self, user_input: str, session_id: str = "default_session",
               callbacks: list | None = None) -> str:
        chain = self.get_conversational_rag_chain(
            streaming=bool(callbacks), callbacks=callbacks
        )
        response = chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": session_id},
                "callbacks": callbacks or [],
            },
        )
        return response["answer"]

    def stream(self, user_input: str, session_id: str = "default_session",
               callbacks: list | None = None):
        chain = self.get_conversational_rag_chain(
            streaming=True, callbacks=callbacks
        )
        for chunk in chain.stream(
            {"input": user_input},
            config={
                "configurable": {"session_id": session_id},
                "callbacks": callbacks or [],
            },
        ):
            if "answer" in chunk:
                yield chunk["answer"]

    def get_chat_history(self, session_id: str = "default_session") -> list:
        if session_id in self.store:
            return self.store[session_id].messages
        return []


# ── Structured Email Generator (uses RAG retriever for context) ───────

class EmailGenerator:
    def __init__(self, model_params: dict, vector_store_retriever):
        self.model_params = model_params
        self.vector_store_retriever = vector_store_retriever

    def parse_query(self, user_query: str, callbacks: list | None = None) -> QueryFilter:
        """Classify intent and extract filter criteria from user query."""
        llm = build_llm(self.model_params, streaming=True, callbacks=callbacks)

        filter_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a query classifier and filter parser for a cold-email system.\n\n"
             "Determine TWO things from the user's message:\n"
             "1. **is_email_request**: Is the user asking to generate/draft/write/compose/send a cold email? "
             "Set True if yes, False if it is a general question.\n"
             "2. **Filters** (only relevant when is_email_request=True):\n"
             "   - company_types: list of company types mentioned. "
             "Recognized types: Startup, Product Based, IT Services, IT Services & Consultancy, "
             "FinTech, GCCs, Unicorn, HealthTech. "
             "Map synonyms: 'service based' -> 'IT Services & Consultancy', "
             "'product companies' -> 'Product Based'. "
             "If MULTIPLE types are mentioned (e.g. 'product based and startup'), return ALL as a list. "
             "None if not mentioned.\n"
             "   - hiring_role: the job role mentioned (e.g. 'AI Engineer', 'Python Developer', "
             "'ML Engineer', 'Data Scientist', 'Software Engineer', 'Software Developer', "
             "'Software Engineer in Test'). None if not mentioned.\n"
             "   - company: a specific company name (e.g. 'Google', 'ABC', 'XYZ'). "
             "Do NOT confuse company-type words like 'startup' with company names. None if not mentioned.\n\n"
             "Examples:\n"
             "- 'draft me a cold email for AI Engineer role in startups' -> "
             "is_email_request=True, company_types=['Startup'], hiring_role='AI Engineer', company=None\n"
             "- 'generate email for product based and startup companies' -> "
             "is_email_request=True, company_types=['Product Based', 'Startup'], hiring_role=None, company=None\n"
             "- 'cold email for AI Engineer at ABC company' -> "
             "is_email_request=True, company_types=None, hiring_role='AI Engineer', company='ABC'\n"
             "- 'write email for python developer in fintech' -> "
             "is_email_request=True, company_types=['FinTech'], hiring_role='Python Developer', company=None\n"
             "- 'draft email for ABC startup company' -> "
             "is_email_request=True, company_types=['Startup'], hiring_role=None, company='ABC'\n"
             "- 'what are my skills?' -> is_email_request=False, all filters None\n"
             "- 'generate cold email for all companies' -> is_email_request=True, all filters None\n"
             "Return None for any filter not mentioned in the query."),
            ("human", "{input}"),
        ])

        structured_llm = llm.with_structured_output(QueryFilter)
        chain = filter_prompt | structured_llm
        return chain.invoke({"input": user_query}, config={"callbacks": callbacks or []})

    def _filter_hr_records(self, hr_records: list[dict], filters: QueryFilter) -> list[dict]:
        filtered = hr_records

        if filters.company_types:
            query_types = [ct.lower().strip() for ct in filters.company_types]
            filtered = [
                r for r in filtered
                if any(ct in str(r.get("Company_Type", "")).lower() for ct in query_types)
            ]

        if filters.hiring_role:
            query_hr = filters.hiring_role.lower().strip()
            filtered = [
                r for r in filtered
                if query_hr in str(r.get("Hiring Role", "")).lower()
            ]

        if filters.company:
            query_co = filters.company.lower().strip()
            filtered = [
                r for r in filtered
                if query_co in str(r.get("Company", "")).lower()
            ]

        return filtered

    def generate_emails(
        self,
        user_query: str,
        hr_records: list[dict],
        filters: QueryFilter | None = None,
        callbacks: list | None = None,
    ) -> tuple[list[tuple[ColdEmail, list[dict]]], QueryFilter]:
        """Generate tailored emails grouped by (Company, Hiring Role)."""
        if not hr_records:
            raise ValueError("No HR contact records provided.")

        # Step 1: Use pre-parsed filters or parse now
        if filters is None:
            filters = self.parse_query(user_query, callbacks=callbacks)

        # Step 2: Filter HR records with progressive fallback
        filtered = self._filter_hr_records(hr_records, filters)

        # Fallback 1: drop hiring_role if no results
        if not filtered and filters.hiring_role:
            relaxed = QueryFilter(
                is_email_request=True,
                company_types=filters.company_types,
                hiring_role=None,
                company=filters.company,
            )
            filtered = self._filter_hr_records(hr_records, relaxed)

        # Fallback 2: drop company_types if still no results
        if not filtered and filters.company_types:
            relaxed = QueryFilter(
                is_email_request=True,
                company_types=None,
                hiring_role=filters.hiring_role,
                company=filters.company,
            )
            filtered = self._filter_hr_records(hr_records, relaxed)

        # Fallback 3: use all records
        if not filtered:
            filtered = hr_records

        # Step 3: Retrieve resume context once (shared across all emails)
        retrieved_docs = self.vector_store_retriever.invoke(user_query)
        resume_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Step 4: One email per contact (personalised greeting per HR person)
        sorted_records = sorted(
            filtered,
            key=lambda r: (r.get("Company", ""), r.get("Hiring Role", ""), r.get("HR Name/Team", "")),
        )

        # Step 5: Generate one tailored email per individual contact
        llm = build_llm(self.model_params, streaming=True, callbacks=callbacks)
        structured_llm = llm.with_structured_output(ColdEmail)

        results: list[tuple[ColdEmail, list[dict]]] = []
        for record in sorted_records:
            company = record.get("Company", "Unknown")
            role = record.get("Hiring Role", "Unknown")
            hr_name = record.get("HR Name/Team", "N/A")
            company_type = record.get("Company_Type", "") or "N/A"

            system_prompt = (
                "You are an expert cold-email copywriter for job seekers.\n\n"
                "Generate exactly ONE professional cold email for the target described below.\n\n"
                "Guidelines:\n"
                "- GREETING RULES (critical):\n"
                "  * If the recipient name looks like a real person's name "
                "(e.g., 'Priya', 'John', 'Rahul Sharma'), use 'Dear <FirstName>,' "
                "(e.g., 'Dear Priya,').\n"
                "  * If the name looks like a team or department "
                "(e.g., 'HR Team', 'Talent Acquisition', 'Recruitment Team'), "
                "use 'Dear Hiring Team,' or 'Dear Talent Acquisition Team,'.\n"
                "- MENTION the specific company name and role in the email body naturally.\n"
                "- NO LINKS: The email must NOT contain any URLs, hyperlinks, portfolio links, "
                "GitHub links, LinkedIn links, or any web addresses. Emails with links will bounce.\n"
                "- Tailor the tone to the company type "
                "(formal for enterprise/GCCs, slightly casual for startups).\n"
                "- Highlight the candidate's relevant skills and experience from their resume.\n"
                "- Keep the body concise (150-250 words).\n"
                "- Include a clear call-to-action (e.g., requesting a brief introductory call).\n"
                "- Do NOT use placeholder brackets like [Company Name].\n"
                "- The signature should only contain the candidate's name, phone, and email.\n\n"
                "Candidate Resume:\n---\n"
                f"{resume_context}\n---\n\n"
                f"Target Company: {company}\n"
                f"Company Type: {company_type}\n"
                f"Hiring Role: {role}\n"
                f"Recipient (HR/TA): {hr_name}"
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query),
            ]
            email = structured_llm.invoke(
                messages,
                config={"callbacks": callbacks or []},
            )
            results.append((email, [record]))

        return results, filters

