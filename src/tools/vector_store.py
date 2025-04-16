from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np

def generate_faiss_index(chunks, model_name="all-MiniLM-L6-v2", index_path="my_index.faiss"):
    # Load embedding model
    model = SentenceTransformer(model_name)

    # Generate embeddings
    embeddings = model.encode(chunks)

    # Create FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Save index
    faiss.write_index(index, index_path)
    print(f"✅ FAISS index saved at: {index_path}")
    return index
