from src.tools.vector_store import generate_faiss_index
from src.tools.text_utils import chunk_text
from src.tools.pdf_loader import load_pdfs_from_folder
import json

papers = load_pdfs_from_folder("data/papers")

all_chunks = []
for name, full_text in papers:
    chunks = chunk_text(full_text)
    all_chunks.extend(chunks)

# Save all chunks to JSON
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

generate_faiss_index(all_chunks)
