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


import os
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

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


