# Interview notes — Premier League RAG Assistant

**Read this thoroughly before putting the project on your CV.** If you can answer the questions at the bottom of this file in your own words, you are ready to defend the project. If you cannot, you are not ready and adding it to your CV is risky.

---

## What this project is, in one sentence

A retrieval-augmented generation (RAG) chatbot that answers questions about the Premier League by retrieving the most relevant chunks from a vector index of official documents and asking an LLM to answer using only those chunks, with citations.

---

## The pipeline, end to end

### 1. Ingestion (one-time, `ingest.py`)

```
PDFs/TXT  →  text extraction  →  chunking  →  embedding  →  FAISS index
```

- **Text extraction**: `pypdf` pulls the text layer page by page. (Scanned PDFs without a text layer would need OCR — a known limitation.)
- **Chunking**: sliding-window character chunks of 800 chars with 150 char overlap. Each chunk keeps a reference to its source file and page so we can cite it later.
- **Embedding**: each chunk → a 768-dimensional vector from Google's `gemini-embedding-001` with `task_type="RETRIEVAL_DOCUMENT"`. Chunks are embedded in batches of 16 with a short delay to stay friendly to free-tier rate limits.
- **Indexing**: vectors are L2-normalised and added to a FAISS `IndexFlatIP` (inner product). On unit-length vectors, inner product equals cosine similarity, and `IndexFlatIP` is the fastest exact search index in FAISS.
- The index plus a parallel list of chunk metadata `(text, source, page)` are saved to `data/cache/`.

### 2. Query time (`rag.py`)

```
question  →  embed  →  top-5 search in FAISS  →  prompt with context  →  LLM answer + citations
```

- **Embed the question** with the same embedding model and dimension used at ingest time, but with `task_type="RETRIEVAL_QUERY"` so the model knows this is a search query (Gemini embeddings are asymmetric — queries and documents get slightly different representations to improve retrieval).
- **Retrieve** the top-5 most similar chunks.
- **Prompt construction**: a system instruction that tells the model to answer ONLY from the context, then a user message containing the context blocks and the question.
- **Generation**: `gemini-2.5-flash-lite` at `temperature=0.1` (low = more factual, less creative).
- The answer is returned along with the source filenames and page numbers, which the UI shows in an expandable "Sources" section.

---

## Why these choices (interview-ready justifications)

| Decision | Why |
|---|---|
| **Gemini over OpenAI** | Free tier with no credit card and generous limits (1,000 requests/day on Flash-Lite, very high quotas on embeddings) made this build-it-yourself friendly. The architecture is provider-agnostic — swapping to OpenAI or Azure OpenAI is a 3-line change in `ingest.py` and `rag.py`. Cost-consciousness in LLM choice is a real engineering concern. |
| **FAISS** over Chroma/Pinecone | Free, local, zero infra for a demo. `IndexFlatIP` is exact (not approximate) so retrieval quality is the ceiling, not the index. Easy to swap for a managed vector DB later — the interface is the same. |
| **`IndexFlatIP` + L2-normalisation** | Exact search, fast for tens of thousands of chunks. Inner product on unit vectors equals cosine similarity, which is the standard for embedding retrieval. Mathematically cleaner and faster than `IndexFlatL2` plus a conversion. |
| **`gemini-embedding-001`** at 768 dims | The model supports 768/1536/3072 dimensions (Matryoshka representation learning). 768 is 4× faster for similarity search than the default 3072 with negligible quality loss for English Q&A. |
| **Asymmetric task types** (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`) | Gemini embeddings are trained asymmetrically — telling the model whether a string is a document being indexed or a query being searched produces better retrieval. A small detail with a real precision/recall benefit. |
| **`gemini-2.5-flash-lite`** for generation | Free tier, fast, more than capable for grounded Q&A. The hard part of RAG is retrieval, not generation; spending more on the chat model rarely fixes a retrieval problem. |
| **Character chunks of 800 with 150 overlap** | Simple and competitive. Overlap reduces the chance of splitting a key sentence. Token-aware or semantic chunking are obvious follow-ups. |
| **`temperature=0.1`** | Q&A is a factual task. Higher temperatures encourage the model to invent. |
| **Strict system instruction** ("only use the context") | The main defence against hallucination in RAG. Combined with low temperature, the model overwhelmingly stays grounded. |
| **Citations in the answer** | Trust and verifiability — users can check sources. Also a soft hallucination check: if the model can't cite, it's making things up. |
| **Streamlit for UI** | Single Python file, no front-end build step, looks reasonable. Right for a demo; production would use a real web stack (FastAPI + React). |

---

## Known limitations (admit these before the interviewer points them out)

1. **Scanned PDFs need OCR** — `pypdf` only reads the text layer. Tesseract or AWS Textract would handle scans.
2. **Character chunking can split sentences** — semantic chunking (`unstructured`, `LlamaIndex`) would improve recall for some queries.
3. **No re-ranking** — the top-5 by cosine similarity isn't always the best 5 for the question. A cross-encoder re-ranker is a well-known next step.
4. **No conversation memory** — each question is independent. Multi-turn would need history-aware retrieval (rewrite the follow-up question into a standalone form before embedding).
5. **No evaluation harness** — I would build one with a hand-labelled set of Q/A pairs and measure precision@k for retrieval and an LLM-as-judge score for answer quality.
6. **No hybrid search** — pure dense retrieval can miss exact-match queries (e.g. specific rule numbers). BM25 + dense (RRF fusion) would help.
7. **Single-tenant, in-memory index** — production would use a managed vector DB (pgvector, Pinecone, Azure AI Search) for persistence and multi-user scale.
8. **Free-tier rate limits** — at ingestion time, I batch and sleep between batches; for a very large corpus a real production system would use the batch embeddings API and proper queue/retry.

---

## How this project maps to the Premier League AI Engineer job description

| JD requirement | How this project demonstrates it |
|---|---|
| "Building AI agents, copilots, or automation workflows using LLMs" | This is a copilot for navigating internal documents — the simplest, most common enterprise LLM use case. |
| "Strong understanding of prompt engineering, RAG architectures" | The whole project is RAG. The system instruction is a deliberate piece of prompt engineering aimed at grounding. |
| "Embeddings, vector databases, and retrieval-augmented generation" | Embedded with Gemini, stored in FAISS, retrieved by cosine similarity, with asymmetric query/document task types. |
| "Evaluate and optimise model performance for real-world use cases" | I can explain the evaluation gap (no test harness yet) and the obvious quality levers: chunk size, top-k, re-ranking, hybrid search. |
| "Experience integrating AI solutions with APIs, databases, and enterprise systems" | Gemini API integration with retry-friendly batching and rate-limit-aware throttling; trivially swappable to OpenAI or Azure OpenAI. |
| "Azure OpenAI" | I built against Gemini because of cost constraints, but the architecture is provider-agnostic — I'd change the client construction and model strings, swap `task_type` for OpenAI's equivalent, and the rest of the code (FAISS, chunking, system prompt) is identical. |
| "Hallucination, data leakage, and misuse with mitigation strategies" | Grounded prompt + low temperature + citations as the standard RAG mitigation set. No user data persists in the index. |
| "Cost control" | A real concern in production LLM systems — choosing Flash-Lite over Pro and 768-dim over 3072-dim embeddings were both deliberate cost choices. |

---

## Likely interview questions and short answers

**"Walk me through what happens when a user asks a question."**
Their text is embedded with `gemini-embedding-001` at 768 dimensions, with task type set to RETRIEVAL_QUERY, then L2-normalised. FAISS does an inner-product search against the index for the top 5 nearest chunks. Those chunks plus the question are sent to `gemini-2.5-flash-lite` with a system instruction that restricts the model to using only the provided context and to citing sources. The answer is returned with the source filenames and page numbers.

**"Why Gemini and not OpenAI?"**
For this project, the deciding factor was free-tier access without billing setup — I wanted to build and iterate without cost pressure. But the architecture is provider-agnostic: the embedding and generation calls are isolated, and switching to OpenAI or Azure OpenAI is a small change in two files. For a production deployment at the Premier League I would default to Azure OpenAI given the rest of the Microsoft 365 stack.

**"Why FAISS and not Pinecone/Chroma/pgvector?"**
For a demo: local, free, zero infrastructure. For production: I'd choose based on scale and ops requirements — pgvector if I already have Postgres, Azure AI Search if I'm on Azure (which fits this role), Pinecone if I want a managed service with strong filtering and metadata search.

**"What are these task_type parameters doing?"**
Gemini embeddings are asymmetric. The model is trained so that a document embedded with `RETRIEVAL_DOCUMENT` and a query embedded with `RETRIEVAL_QUERY` end up closer in vector space when they actually match, than they would if both were embedded the same way. It's a small detail that improves precision and recall without any extra cost.

**"How do you prevent hallucinations?"**
Three layers: (1) strict system instruction telling the model to use only the context and to say "I don't know" otherwise, (2) low temperature, (3) require citations — if it can't cite, it's likely inventing. The deeper answer is also retrieval quality: if you retrieve the right context, the model has less reason to invent.

**"What would you change for production?"**
Re-ranking, evaluation harness, conversation memory, hybrid search, Azure OpenAI, proper vector DB, logging and cost monitoring, rate limiting and retries, auth, and an actual front-end. Plus chunk-quality improvements (semantic chunking, metadata extraction).

**"How would you evaluate retrieval quality?"**
Build a small labelled set: questions, expected source chunks, and gold answers. Measure recall@5 and precision@5 for retrieval, and use LLM-as-judge or human review for answer quality. Iterate on chunking, embedding model and top-k based on those numbers.

**"What's the cost of running this?"**
Free, on Gemini's free tier — 1,000 chat requests per day on Flash-Lite and very high embedding quotas. For comparison, the equivalent OpenAI build with `text-embedding-3-small` and `gpt-4o-mini` would cost roughly $0.10–$0.20 to ingest a 600-page handbook and about $0.001 per question — still small, but not zero.

---

## What you must do to actually own this project

1. **Set it up and run it locally.** Follow the README step by step. Get to the point where you can ask a question in the Streamlit UI and see a cited answer.
2. **Read every line of `ingest.py` and `rag.py`.** Both files are under 200 lines and heavily commented. If a line confuses you, look it up.
3. **Break something on purpose**, then fix it. Try: change `TOP_K` to 1 and see how answers get worse; change the system prompt to be permissive and watch hallucination appear; change `temperature` to 1.5 and see the model get creative.
4. **Make at least one substantive change of your own** — a feature, a different chunking strategy, a different embedding dimension, a small evaluation script. This is the thing you'll get most excited talking about, and it's what makes the project yours and not a tutorial.
5. **Take a screenshot of it working** and put it in the README.
6. **Push to GitHub** with a clean commit history (small, named commits, not one big "initial commit").

Once that's all true, this project is genuinely yours. Add it to your CV. Mention it in your cover letter. Be ready to discuss any line of it.
