from chromadb import PersistentClient
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

class RAGConfig:
    DB_PATH = "./chroma_db"
    EMBEDDING_MODEL = "all-mpnet-base-v2"
    K = 3

def get_chroma_client():
    return PersistentClient(path=RAGConfig.DB_PATH)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=RAGConfig.EMBEDDING_MODEL)

def get_retriever(collection: str) -> VectorStoreRetriever:
    """
 Creates and returns a LangChain retriever for the specified ChromaDB collection.
    Args:
        collection: Name of the collection (e.g., 'core_db', 'user_{user_id}').
    Returns:
        A configured VectorStoreRetriever instance for the collection.
    Raises:
        Exception: If there is an error initializing ChromaDB or the retriever.
    """
    try:
        client = get_chroma_client()
        embeddings = get_embeddings()
        db = Chroma(client=client, collection_name=collection, embedding_function=embeddings)
        return db.as_retriever(search_type="similarity", search_kwargs={"k": RAGConfig.K})
    
    except Exception as e:
        print(f"Error creating retriever for collection '{collection}': {e}")
        raise
        