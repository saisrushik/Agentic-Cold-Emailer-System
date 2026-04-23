from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from pydantic import BaseModel, Field
from data_injection.excel_csv import ExcelCsvReader
from rag_pipeline.vector_store import VectorStore

import os
from dotenv import load_dotenv


load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_TRACING"] = "true"


# # model = ChatOpenAI(
# #     model="...",
# #     temperature=0,
# #     max_tokens=None,
# #     timeout=None,
# #     max_retries=2,
# #     api_key = ""
# # )


class RagChain:
    def __init__(self, model_params, vector_store_retriever):
        # self.llm_provider = llm_provider
        self.model_params = model_params
        self.hr_details = []
        self.vector_store_retriever = vector_store_retriever

    

    def get_hr_contacts(self):
        excel = ExcelCsvReader("../Data/email.csv")
        self.hr_details = excel.read_csv().to_dict(orient='records')

    def get_rag_chain(self):
        model = self.model_params['model']
        temperature = self.model_params['temperature']
        max_tokens = self.model_params['max_tokens']
        max_retries = self.model_params['max_retries']

        if model == "groq":
            llm = ChatGroq(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
        elif model == "openai":
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
        elif model == "ollama":
            llm = ChatOllama(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
        
        #Contextualize conversation
        contextualize_conversation_system_prompt=(
            "Given a chat history and the latest user question"
            "which might be referring to previous questions or context in the chat history, "
            "formulate a standalone summary of the conversation which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        #contextualize conversation prompt
        contextualize_conversation_prompt = ChatPromptTemplate.from_messages(

            [
                ("system", contextualize_conversation_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, self.vector_store_retriever, contextualize_conversation_prompt)

        #query prompt
        query_prompt = (
            """
                You are an helpful assistant for generating the cold emails for HR Professionals or the talent acquisition team. 
                Your goal is to write a highly personalized, non-spammy cold email using specific context.
                Use the following pieces of retrieved context to answer the user query. 
                \n\n
                {context}
            """
        )

        rag_chain_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", query_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, rag_chain_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        return rag_chain

    

