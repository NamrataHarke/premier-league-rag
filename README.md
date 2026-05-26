# Premier League Handbook RAG Assistant ⚽

A retrieval-augmented generation (RAG) chatbot that answers questions about Premier League rules and regulations using only the official Handbook as a knowledge source — every answer is grounded in the documents and cites its sources.

Built as a practical demonstration of production-style LLM patterns: document ingestion, chunking, embedding, vector retrieval, and grounded generation.

## What it does

Drop one or more PDFs into `data/`, run the ingest script once, then chat with the documents through a Streamlit UI. Each answer cites the source file and page.

## Architecture

```
PDFs/TXT  ─►  ingest.py  ─►  FAISS index + chunk metadata
                                       ▲
                                       │ retrieve top-5
              ┌────────────┐           │
   User  ─►   │ Streamlit  │  ─►  rag.py  ─►  Gemini  ─►  Answer + citations
              └────────────┘
```

- **Embeddings**: Google `gemini-embedding-001` (768 dims, free tier)
- **Vector store**: FAISS `IndexFlatIP` over L2-normalised vectors (exact cosine search)
- **Generation**: Google `gemini-2.5-flash-lite` at `temperature=0.1` (free tier)
- **UI**: Streamlit chat interface with source citations

The architecture is provider-agnostic — the embedding and generation calls are isolated in `ingest.py` and `rag.py` and could be swapped for OpenAI or Azure OpenAI with a ~3-line change in each file.

## Setup

```bash
# 1. Clone and create a virtual environment
git clone <this-repo>
cd premier-league-rag
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Gemini API key
cp .env.example .env                # then edit .env with your key
```

Get a **free** API key from <https://aistudio.google.com/app/apikey> — no credit card, generous free tier (1,000 requests per day on `gemini-2.5-flash-lite` and very high quotas on embeddings).

## Run

```bash
# 4. Drop documents into data/ (a small sample text file is included so the
#    project runs out of the box). The official Premier League Handbook is
#    publicly available from the Premier League website.

# 5. Build the vector index (one-off)
python ingest.py

# 6. Launch the chat UI
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Example questions

- "How many clubs are in the Premier League?"
- "How does promotion and relegation work?"
- "What is VAR used for?"
- "How does financial regulation work in the Premier League?"

## Cost

Completely free on Google's Gemini free tier for normal demo usage. Limits are 15 requests per minute and 1,000 per day on `gemini-2.5-flash-lite`, plus very generous quotas on `gemini-embedding-001`.

## Design decisions

See [`INTERVIEW_NOTES.md`](INTERVIEW_NOTES.md) for the full design rationale: why FAISS, why `IndexFlatIP`, why these models, known limitations, and a list of production improvements.

## Possible extensions

- Swap to OpenAI or Azure OpenAI (≈ 3-line change in `ingest.py` and `rag.py`)
- Re-ranking with a cross-encoder
- Hybrid search (BM25 + dense, RRF fusion)
- Conversation memory across turns
- Evaluation harness (precision@k, LLM-as-judge)
- Replace FAISS with a managed vector DB (pgvector, Pinecone, Azure AI Search)

## Repository structure

```
premier-league-rag/
├── README.md
├── INTERVIEW_NOTES.md      # Design decisions and limitations
├── requirements.txt
├── .env.example
├── ingest.py               # Documents → chunks → embeddings → FAISS
├── rag.py                  # Retrieve top-k + generate grounded answer
├── app.py                  # Streamlit chat UI
└── data/
    ├── README.md           # Where to put your documents
    └── sample_premier_league_info.txt   # Sample doc so the app runs out of the box
```

## Author

Namrata Mahendra Harke — MSc Artificial Intelligence, University of Sheffield.
