import os
from pathlib import Path
from PyPDF2 import PdfReader
from typing import List, Tuple


def load_pdfs_from_folder(folder_path: str) -> List[Tuple[str, str]]:
    pdf_texts = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = Path(folder_path) / file
            try:
                reader = PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text.replace('\n', ' ').strip()
                pdf_texts.append((file, text))
            except Exception as e:
                print(f"❌ Failed to load {file}: {e}")
    return pdf_texts


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


if __name__ == "__main__":
    folder = "data/paper"  # adjust if needed
    papers = load_pdfs_from_folder(folder)
    print(f"📄 Loaded {len(papers)} PDF(s).")

    for name, full_text in papers:
        chunks = chunk_text(full_text)
        print(f"\n🧠 {name} split into {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks[:2]):  # just showing the first 2 for now
            print(f"  Chunk {i+1}:\n{chunk[:300]}...\n")
