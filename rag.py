"""
Core RAG (retrieval-augmented generation) logic.

For each user question:
  1. Embed the question with the same model used at ingest time
  2. Retrieve the top-k most similar chunks from the FAISS index
  3. Ask the LLM to answer using ONLY those chunks, with citations

This module exposes:
  - load_index() -> (faiss_index, chunks_list)
  - answer(query, index, chunks) -> {"answer": str, "sources": [...]}
"""

import os
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ---- Configuration ---------------------------------------------------------
CACHE_DIR = Path("data/cache")

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768  # must match ingest.py
CHAT_MODEL = "gemini-2.5-flash-lite"  # free tier, fast, good for grounded Q&A
TOP_K = 5  # number of chunks retrieved per question
MAX_RETRIES = 3  # retry transient server errors (e.g. 503 UNAVAILABLE)
# ---------------------------------------------------------------------------

# The system prompt is deliberately strict about grounding. This is the main
# defence against hallucination in a RAG system.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about \
the Premier League using ONLY the provided context passages.

Rules:
- Answer based on the context only. Do not use outside knowledge.
- If the context does not contain the answer, say:
  "I could not find that in the provided documents."
- After each factual claim, cite the source in square brackets,
  e.g. [Handbook-2024-25.pdf p.42].
- Be concise and factual; do not invent details."""


def load_index():
    """Load the FAISS index and chunk metadata produced by ingest.py."""
    index_path = CACHE_DIR / "index.faiss"
    chunks_path = CACHE_DIR / "chunks.pkl"
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            "No index found. Run `python ingest.py` first."
        )
    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string.

    Uses task_type='RETRIEVAL_QUERY' to tell the model this is a search query,
    which produces a slightly different embedding than RETRIEVAL_DOCUMENT.
    Must use the same model and dimension as ingestion.
    """
    result = _with_retry(
        lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=query,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBED_DIM,
                task_type="RETRIEVAL_QUERY",
            ),
        )
    )
    vec = np.array([result.embeddings[0].values], dtype="float32")
    # Normalise so inner-product search == cosine similarity
    faiss.normalize_L2(vec)
    return vec


def retrieve(query: str, index, chunks, k: int = TOP_K):
    """Return the top-k (chunk_tuple, similarity_score) pairs for a query."""
    q_vec = embed_query(query)
    scores, indices = index.search(q_vec, k)
    return [
        (chunks[idx], float(scores[0][rank]))
        for rank, idx in enumerate(indices[0])
    ]


def format_context(retrieved) -> str:
    """Format retrieved chunks into a single string the LLM can read."""
    blocks = []
    for (text, source, location), score in retrieved:
        header = f"[Source: {source} {location} | similarity: {score:.3f}]"
        blocks.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(blocks)


def _with_retry(api_call):
    """Run a Gemini API call with simple exponential-backoff retry.

    The Gemini free tier occasionally returns 503 UNAVAILABLE during demand
    spikes. Production LLM systems always wrap API calls in retry logic for
    transient server errors. We retry up to MAX_RETRIES times with backoff
    of 1s, 2s, 4s.
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return api_call()
        except genai_errors.ServerError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(
                    f"Gemini API returned {e.code}; "
                    f"retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})..."
                )
                time.sleep(wait)
    raise last_error


def answer(query: str, index=None, chunks=None) -> dict:
    """Answer a question using RAG.

    Returns a dict:
      {
        "answer": "<the LLM's grounded answer with citations>",
        "sources": [(filename, location, similarity), ...]
      }
    """
    if index is None or chunks is None:
        index, chunks = load_index()

    retrieved = retrieve(query, index, chunks)
    context = format_context(retrieved)

    response = _with_retry(
        lambda: client.models.generate_content(
            model=CHAT_MODEL,
            contents=f"Context:\n{context}\n\nQuestion: {query}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,  # low temperature = less creative, more factual
            ),
        )
    )

    return {
        "answer": response.text,
        "sources": [
            (src, loc, score) for (_, src, loc), score in retrieved
        ],
    }


if __name__ == "__main__":
    # Simple CLI for testing without the Streamlit UI
    print("Loading index...")
    index, chunks = load_index()
    print(f"Loaded {len(chunks)} chunks. Ask a question (or 'quit').\n")
    while True:
        q = input("Q: ").strip()
        if not q or q.lower() in ("quit", "exit", "q"):
            break
        result = answer(q, index, chunks)
        print(f"\nA: {result['answer']}\n")
        print("Sources:")
        for src, loc, score in result["sources"]:
            print(f"  - {src} {loc}  (similarity: {score:.3f})")
        print()