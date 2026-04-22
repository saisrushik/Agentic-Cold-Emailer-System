from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")


class VectoreStore:
    def __init__(self, docs):
        self.docs = docs
        self.vectore_store_db = None
    
    
    def create_vectore_store(self):

        #splitter
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=1000,
            chunk_overlap=200
        )

        #splitting documents
        splits = splitter.split_documents(self.docs)

        # embeddings
        # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        embeddings = OllamaEmbeddings(model="all-MiniLM-L6-v2")
        
        # vectore store
        # Make sure to set PINECONE_API_KEY in your .env file
        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = os.environ.get("PINECONE_INDEX_NAME", "resume-embeddings")

        try:
            if api_key is None:
                raise ValueError("PINECONE_API_KEY not found in environment variables")
            if index_name is None:
                raise ValueError("PINECONE_INDEX_NAME not found in environment variables")
        except Exception as e:
            print(e)
            return None
        
        pc = Pinecone(api_key=api_key)

        #delete old index
        if pc.has_index(index_name):
            pc.delete_index(index_name)
            print(f"Deleted old index {index_name}")
        
        # Create pinecone index if it doesn't exist
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        
        # Initialize Pinecone client and clear existing vectors
        index = pc.Index(index_name)
            
        # Delete all existing vectors in the default namespace
        try:
            stats = index.describe_index_stats()
            if stats.total_vector_count > 0:
                # The index has data, so we can safely delete
                index.delete(delete_all=True, namespace="")
                print(f"Successfully cleared existing embeddings from index: {index_name}")
            else:
                print(f"Index '{index_name}' is already empty. No need to clear.")
        except Exception as e:
            print(f"Note: Could not clear index: {e}")

        self.vectore_store_db = PineconeVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            index_name=index_name
        )

        return self.vectore_store_db
        
    
    def similarity_search_with_score(self, query, k=5, score_threshold=0.5):
        try:
            if self.vectore_store_db is None:
                raise ValueError("Vector store not initialized. Please create it first.")
            
            # perform similarity search
            results = self.vectore_store_db.similarity_search_with_score(query, k=k, score_threshold=score_threshold)
            return results    

        except Exception as e:
            print(e)
            return None