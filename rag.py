# rag.py
import os
from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from pinecone import Pinecone # Import the main Pinecone client class

print("Loading environment variables for rag.py...")
load_dotenv() # Load env vars early

# --- Initialize Pinecone Client ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set.")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME environment variable not set.")

print(f"Initializing Pinecone client for index: {PINECONE_INDEX_NAME}")
# Initialize the client globally (or create instance as needed)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
print("Pinecone client initialized.")

class RAGConfig:
    EMBEDDING_MODEL = "text-embedding-3-small"
    K = 3

def get_embeddings():
    return OpenAIEmbeddings(
        model=RAGConfig.EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

def get_vector_store(namespace: str = None) -> PineconeVectorStore:
    """Gets the PineconeVectorStore instance, assuming client is initialized."""
    embeddings = get_embeddings()

    # PineconeVectorStore uses the globally initialized client or finds it implicitly
    # It requires index_name passed here.
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME, # Pass index name here
        embedding=embeddings,
        namespace=namespace # Use Pinecone namespaces for separation
    )
    return vector_store

# --- get_retriever function remains the SAME as the previous Pinecone version ---
def get_retriever(collection_name: str) -> VectorStoreRetriever:
    """
    Creates and returns a LangChain retriever for PineconeVectorStore,
    using namespaces to simulate collections. Namespace = collection_name.
    """
    try:
        namespace = collection_name
        vector_store = get_vector_store(namespace=namespace)
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': RAGConfig.K}
        )
    except Exception as e:
        print(f"Error creating Pinecone retriever for namespace '{collection_name}': {e}")
        raise