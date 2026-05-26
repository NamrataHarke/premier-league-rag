"""
Ingest pipeline for the Premier League Handbook RAG assistant.

What this script does:
  1. Loads every PDF and .txt file in ./data/
  2. Splits each document into overlapping character chunks
  3. Embeds each chunk with Google's gemini-embedding-001
  4. Builds a FAISS vector index and saves it to ./data/cache/

Run after dropping documents into ./data/:
    python ingest.py
"""

import os
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

# Load GEMINI_API_KEY from .env
load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ---- Configuration ---------------------------------------------------------
DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"

EMBED_MODEL = "gemini-embedding-001"  # free tier, generous limits
EMBED_DIM = 768                        # 768/1536/3072 supported; smaller = faster
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between chunks to avoid splitting sentences
BATCH_SIZE = 16        # chunks per API call (kept small to stay under rate limits)
BATCH_DELAY = 1.0      # seconds between batches (free tier rate-limit safety)
# ---------------------------------------------------------------------------


def load_documents(data_dir: Path):
    """Yield (text, source_filename, page_or_section) for every page or text file."""
    files = sorted(list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.txt")))
    files = [f for f in files if f.parent.name != "cache"]
    if not files:
        raise FileNotFoundError(
            f"No .pdf or .txt files found in {data_dir}/. "
            "Drop your documents there first (see data/README.md)."
        )

    for path in files:
        print(f"Reading {path.name}...")
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    yield text, path.name, f"p.{page_num}"
        else:  # .txt
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                yield text, path.name, "text"


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Sliding-window character chunking.

    Simple but a strong baseline for RAG. More sophisticated options
    (sentence- or semantic-boundary chunking) are listed in INTERVIEW_NOTES.md
    as obvious follow-up improvements.
    """
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def embed_batch(texts: list) -> np.ndarray:
    """Call the Gemini embeddings API for a batch of strings.

    Returns a (n, d) float32 numpy array where n is the number of texts and
    d is the embedding dimension (768 here).
    """
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBED_DIM,
            task_type="RETRIEVAL_DOCUMENT",  # tells the model these are documents to be retrieved
        ),
    )
    vectors = [emb.values for emb in result.embeddings]
    return np.array(vectors, dtype="float32")


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and chunk every document
    all_chunks = []  # list of (text, source_filename, location_label)
    for page_text, source, location in load_documents(DATA_DIR):
        for chunk in chunk_text(page_text):
            all_chunks.append((chunk, source, location))
    print(f"\nTotal chunks: {len(all_chunks)}")

    if not all_chunks:
        print("No text was extracted. Are your PDFs scanned images (no text layer)?")
        return

    # 2. Embed every chunk in small batches with a short delay between batches
    texts = [c[0] for c in all_chunks]
    vectors_batches = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        print(f"Embedding {i + len(batch)}/{len(texts)} chunks...")
        vectors_batches.append(embed_batch(batch))
        if i + BATCH_SIZE < len(texts):
            time.sleep(BATCH_DELAY)  # gentle on free-tier rate limits
    vectors = np.vstack(vectors_batches)

    # 3. Build the FAISS index.
    #    L2-normalising the vectors and using IndexFlatIP (inner product)
    #    is equivalent to cosine similarity, and is the fastest exact index
    #    in FAISS. Good enough for tens of thousands of chunks.
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # 4. Save the index and the chunk metadata side by side
    faiss.write_index(index, str(CACHE_DIR / "index.faiss"))
    with open(CACHE_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\nDone. Saved index ({len(all_chunks)} chunks) to {CACHE_DIR}/")
    print("Next step: streamlit run app.py")


if __name__ == "__main__":
    main()
