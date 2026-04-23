from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from pydantic import BaseModel, Field
from typing import Optional
from data_injection.excel_csv import ExcelCsvReader

import os
from dotenv import load_dotenv


load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_TRACING"] = "true"


# ── Pydantic Structured Output Models ─────────────────────────────────

class ColdEmail(BaseModel):
    subject: str = Field(description="Email subject line — concise and attention-grabbing")
    greeting: str = Field(description="Personalized greeting addressing the HR/TA person by name")
    body: str = Field(description="Main email body — professional, non-spammy, and tailored to the company and role")
    closing: str = Field(description="Professional closing statement with a call to action")
    signature: str = Field(description="Email signature with candidate name")


class EmailBatch(BaseModel):
    emails: list[ColdEmail] = Field(description="List of generated cold emails, one per HR contact")


class QueryFilter(BaseModel):
    company_type: Optional[str] = Field(
        default=None,
        description="Company type filter extracted from the user query (e.g., Startup, Product Based, FinTech, GCCs, etc.). None if not specified.",
    )
    hiring_role: Optional[str] = Field(
        default=None,
        description="Hiring role filter extracted from the user query (e.g., AI Engineer, Python Developer, etc.). None if not specified.",
    )
    company: Optional[str] = Field(
        default=None,
        description="Specific company name filter extracted from the user query. None if not specified.",
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
            "- Adapt the language based on the company type (e.g., more formal for enterprise, "
            "slightly casual for startups).\n\n"
            "Retrieved Resume Context:\n"
            "---\n"
            "{context}\n"
            "---\n\n"
            "Use the resume context above to craft emails. "
            "If the user provides HR contact details, generate one email per contact. "
            "Answer any follow-up questions using the resume context and chat history."
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

    def _parse_query_filters(self, user_query: str, callbacks: list | None = None) -> QueryFilter:
        llm = build_llm(self.model_params, callbacks=callbacks)

        filter_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a query parser. Extract filter criteria from the user's request. "
             "The user wants to send cold emails to specific HR contacts. "
             "Extract any company type, hiring role, or company name they mention. "
             "Use fuzzy/semantic matching — for example:\n"
             "- 'startup companies' -> company_type='Startup'\n"
             "- 'AI Engineer roles' -> hiring_role='AI Engineer'\n"
             "- 'product based' -> company_type='Product Based'\n"
             "- 'fintech' -> company_type='FinTech'\n"
             "Return None for any field not mentioned in the query."),
            ("human", "{input}"),
        ])

        structured_llm = llm.with_structured_output(QueryFilter)
        chain = filter_prompt | structured_llm
        return chain.invoke({"input": user_query}, config={"callbacks": callbacks or []})

    def _filter_hr_records(self, hr_records: list[dict], filters: QueryFilter) -> list[dict]:
        filtered = hr_records

        if filters.company_type:
            query_ct = filters.company_type.lower().strip()
            filtered = [
                r for r in filtered
                if query_ct in str(r.get("Company_Type", "")).lower()
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

    def generate_email_previews(
        self,
        user_query: str,
        hr_records: list[dict],
        callbacks: list | None = None,
    ) -> tuple[EmailBatch, list[dict], QueryFilter]:
        if not hr_records:
            raise ValueError("No HR contact records provided.")

        # Step 1: Parse user query to extract filter criteria
        filters = self._parse_query_filters(user_query, callbacks=callbacks)

        # Step 2: Filter HR records based on extracted criteria
        filtered_records = self._filter_hr_records(hr_records, filters)

        if not filtered_records:
            raise ValueError(
                f"No HR contacts match the filter — "
                f"Company Type: '{filters.company_type or 'any'}', "
                f"Hiring Role: '{filters.hiring_role or 'any'}', "
                f"Company: '{filters.company or 'any'}'. "
                f"Check your email.csv data."
            )

        llm = build_llm(self.model_params, callbacks=callbacks)

        # Step 3: Retrieve relevant resume chunks via the vector store
        retrieved_docs = self.vector_store_retriever.invoke(user_query)
        resume_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        hr_details_block = "\n".join(
            f"- HR/TA Name: {r.get('HR Name/Team', 'N/A')}, "
            f"Email: {r.get('Email', 'N/A')}, "
            f"Company: {r.get('Company', 'N/A')}, "
            f"Company Type: {r.get('Company_Type', 'N/A')}, "
            f"Hiring Role: {r.get('Hiring Role', 'N/A')}"
            for r in filtered_records
        )

        system_prompt = (
            "You are an expert cold-email copywriter for job seekers. "
            "Your task is to generate professional, concise, and personalized cold emails "
            "that a candidate can send to HR professionals or Talent Acquisition teams.\n\n"
            "Guidelines:\n"
            "- Each email must be tailored to the specific HR contact, company, company type, and hiring role.\n"
            "- The tone should be professional, confident, and non-spammy.\n"
            "- Keep emails concise (150-250 words for the body).\n"
            "- Highlight the candidate's relevant skills and experience from their resume.\n"
            "- Include a clear call-to-action (e.g., requesting a brief call or meeting).\n"
            "- Do NOT use generic templates — each email should feel unique and genuine.\n"
            "- Adapt the language based on the company type (e.g., more formal for enterprise, "
            "slightly casual for startups).\n\n"
            "Candidate Resume Context:\n"
            "---\n"
            f"{resume_context}\n"
            "---\n\n"
            "HR / TA Contacts to email:\n"
            f"{hr_details_block}\n\n"
            "Generate one cold email for EACH HR contact listed above."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        structured_llm = llm.with_structured_output(EmailBatch)
        chain = prompt | structured_llm
        result = chain.invoke({"input": user_query}, config={"callbacks": callbacks or []})

        return result, filtered_records, filters

