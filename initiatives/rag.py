import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer #specific version
import pickle

model = SentenceTransformer('all-mpnet-base-v2')

def update_vector_store(user_id, new_file_text=""):
    index_file = f"uploads/user_{user_id}_index.faiss"
    texts_file = f"uploads/user_{user_id}_texts.pkl"
    
    # Load existing or start fresh
    if os.path.exists(index_file) and os.path.exists(texts_file):
        index = faiss.read_index(index_file)
        with open(texts_file, "rb") as f:
            texts = pickle.load(f)
    else:
        index = None
        texts = []

    # Only process new file text
    if new_file_text:
        new_chunks = [chunk.strip() for chunk in new_file_text.split(".") if chunk.strip()]
        if new_chunks:
            new_embeddings = model.encode(new_chunks)
            texts.extend(new_chunks)
            if index is None:
                dimension = new_embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
            index.add(np.array(new_embeddings))

            # Save updated index
            faiss.write_index(index, index_file)
            with open(texts_file, "wb") as f:
                pickle.dump(texts, f)
    
    return index, texts

def get_rag_context(summary, user_id):
    index_file = f"uploads/user_{user_id}_index.faiss"
    texts_file = f"uploads/user_{user_id}_texts.pkl"
    
    if not (os.path.exists(index_file) and os.path.exists(texts_file)):
        return ""
    
    index = faiss.read_index(index_file)
    with open(texts_file, "rb") as f:
        texts = pickle.load(f)
    
    if not index or not texts:
        return ""
    
    query_embedding = model.encode([summary])[0]
    distances, indices = index.search(np.array([query_embedding]), 3)
    top_chunks = [texts[idx] for idx in indices[0]]
    return " ".join(top_chunks)