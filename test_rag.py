# test_rag.py (Pinecone Version)
import os
import sys
from dotenv import load_dotenv
import time
import traceback

# --- Setup Path (if needed) ---
# Adjust if rag.py is not in the same folder or findable via PYTHONPATH
# ... (same path setup logic as before if necessary) ...
# --- End Setup Path ---

# --- Import from your project ---
try:
    # Import necessary components from the *updated* rag.py for Pinecone
    from rag import (
        get_embeddings,
        get_vector_store, # This now returns PineconeVectorStore
        get_retriever,    # This now uses Pinecone namespaces
        RAGConfig         # Contains PINECONE_INDEX_NAME now
        # pinecone_client is initialized inside rag.py now
    )
    # Also import the base Document class if needed for dummy data
    from langchain_core.documents import Document
except ImportError as e:
    print(f"Error importing from rag.py: {e}")
    print("Make sure rag.py is updated for Pinecone and in the Python path.")
    sys.exit(1)

# --- Test Configuration ---
print("Loading environment variables from .env file...")
load_dotenv()

# Check required environment variables for Pinecone
required_vars = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
if not all(os.getenv(var) for var in required_vars):
    print(f"Error: Missing required environment variables: {required_vars}")
    print("Ensure you have a .env file with PINECONE_API_KEY and PINECONE_INDEX_NAME.")
    sys.exit(1)

print("Pinecone Index Name:", os.getenv("PINECONE_INDEX_NAME"))

# --- Helper Function to Insert Test Data (Optional - Use with caution) ---
# It's generally better to test with data uploaded via your application.
def insert_test_data_pinecone(namespace="core_db", user_id="test-user-id-pinecone"):
    """Inserts a single dummy document into a Pinecone namespace."""
    print("\n--- Attempting to Insert Dummy Data into Pinecone ---")
    try:
        # Get the vector store configured for the specific namespace
        vector_store = get_vector_store(namespace=namespace)

        dummy_doc = Document(
            page_content=f"This is a Pinecone test document for the {namespace} namespace about solar panel efficiency.",
            metadata={
                "filename": "dummy_pinecone_test.txt",
                "user_id": user_id,
                "original_collection": namespace # Match metadata structure
            }
        )
        print(f"Adding document to Pinecone index '{RAGConfig.PINECONE_INDEX_NAME}', namespace '{namespace}'...")
        vector_store.add_documents([dummy_doc]) # Add the document
        print(f"Dummy document potentially added to namespace '{namespace}'.")
        # Pinecone indexing can take a short while
        print("Pausing for 5 seconds for indexing...")
        time.sleep(5)
        print("--- Dummy Data Insertion Attempt Finished ---")
    except Exception as e:
        print(f"---! Error inserting dummy data into Pinecone !---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("-" * 30)
        traceback.print_exc()


# --- Main Test Function ---
def run_retrieval_test():
    """Tests the retriever functionality with Pinecone."""
    print("\n--- Testing Pinecone Retriever ---")

    # --- Test Parameters ---
    test_query = "solar panel efficiency" # Query likely to match dummy data if inserted
    # *** IMPORTANT: Set this to a NAMESPACE that SHOULD have data ***
    # E.g., 'core_db' or 'user_your-user-id' from data uploaded via your app
    test_namespace = "core_db"

    print(f"Test Query: '{test_query}'")
    print(f"Target Namespace: '{test_namespace}'")
    print(f"Retriever K: {RAGConfig.K}")
    print(f"Using Pinecone Index: '{RAGConfig.PINECONE_INDEX_NAME}'")
    print("-" * 20)

    try:
        # 1. Get the retriever instance for the specific namespace
        print(f"Instantiating retriever for namespace: '{test_namespace}'...")
        retriever = get_retriever(test_namespace) # Uses the namespace
        print("Retriever instantiated.")
        # Note: Pinecone retriever might not expose filter in search_kwargs if using namespace only
        print(f"Retriever Search Kwargs: {retriever.search_kwargs}")

        # 2. Invoke the retriever
        print("\nInvoking retriever.invoke(test_query)...")
        results = retriever.invoke(test_query)

        print("\n--- Retrieval Results ---")
        if results:
            print(f"Retrieved {len(results)} documents:")
            for i, doc in enumerate(results):
                print(f"\nDocument {i+1}:")
                print(f"  Page Content: {doc.page_content[:150]}...") # Show snippet
                print(f"  Metadata: {doc.metadata}")
                # Pinecone might add a score, check if needed:
                # if hasattr(doc, 'score'): print(f"  Score: {doc.score}")
        else:
            print("No documents retrieved.")
            print(f"(Check if data exists in Pinecone index '{RAGConfig.PINECONE_INDEX_NAME}' within namespace '{test_namespace}' matching the query)")

    except Exception as e:
        print("\n---! Retrieval Failed !---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("\n--- Traceback ---")
        traceback.print_exc() # Print full traceback for detailed debugging

    print("\n--- Test Script Finished ---")


# --- Script Execution ---
if __name__ == "__main__":

    # --- OPTIONAL: Insert data before testing ---
    # Set to True ONLY if you are sure no relevant data exists from app uploads.
    INSERT_DATA_FIRST = True
    if INSERT_DATA_FIRST:
        # Make sure the namespace matches 'test_namespace' in run_retrieval_test
        insert_test_data_pinecone(namespace="core_db")

    # --- Run the main test ---
    run_retrieval_test()