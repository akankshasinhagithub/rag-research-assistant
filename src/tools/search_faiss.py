from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

def search_index(query, index_path="my_index.faiss", model_name="all-MiniLM-L6-v2", top_k=5):
    # Load embedding model
    model = SentenceTransformer(model_name)
    
    # Create embedding for the query
    query_embedding = model.encode([query])
    
    # Load FAISS index
    index = faiss.read_index(index_path)
    
    # Perform similarity search
    distances, indices = index.search(np.array(query_embedding), top_k)
    
    return distances, indices

def get_results(indices, chunk_path="chunks.json"):
    # Load saved text chunks
    with open(chunk_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Return top-k matched chunks
    return [chunks[i] for i in indices[0]]

# Optional quick test
if __name__ == "__main__":
    query = "What is LangChain?"
    distances, indices = search_index(query)
    results = get_results(indices)

    for i, res in enumerate(results):
        print(f"\n🔍 Result {i+1}:\n{res}\n{'-'*60}")



