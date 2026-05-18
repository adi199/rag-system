# RAG System

A document question-answering system powered by **Supabase + pgvector**, **LlamaParse**, and **Groq**.

Drop PDFs into `data/`, index them, and ask questions in a chat UI.

<!-- MEDIA: screenshot of the Streamlit chat interface -->

---

## Quickstart

**Prerequisites:** [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd rag-system

# 2. Copy and fill in your API keys
cp .env.example .env

# 3. Drop your PDFs into data/, then run the full setup
uv run python manage.py setup

# 4. Launch the app
uv run streamlit run app.py
```

---

## Workflow

```
data/*.pdf  →  parse  →  data/*.md  →  ingest  →  Supabase  →  app.py
```

| Command | What it does |
|---|---|
| `manage.py setup` | First-time setup: init DB → parse PDFs → ingest |
| `manage.py reindex` | Add new PDFs without resetting the database |
| `manage.py init-db` | Re-deploy schema and `hybrid_search()` RPC only |
| `manage.py parse` | Re-parse PDFs → Markdown only |
| `manage.py ingest` | Re-embed and store Markdown → Supabase only |

```bash
# Adding new documents to an existing deployment:
cp new_docs/*.pdf data/
uv run python manage.py reindex
```

---

## System Design

<!-- MEDIA: architecture diagram — PDF → LlamaParse → Supabase pgvector → Streamlit -->

### Ingestion Pipeline

1. **Parse** — [LlamaParse](https://cloud.llamaindex.ai) converts each PDF page to structured Markdown with a `Context:` summary block injected at the top of every page.
2. **Chunk** — [semchunk](https://github.com/umarbutler/semchunk) splits each page semantically, respecting the token budget of the embedding model.
3. **Embed** — `all-MiniLM-L6-v2` (SentenceTransformers) generates 384-dimensional vectors.
4. **Store** — Chunks are batch-inserted into Supabase Postgres with both an embedding column (`vector(384)`) and a generated `tsvector` column for full-text search.

### Retrieval Pipeline

<!-- MEDIA: short screen recording of a query being answered with citations -->

1. **Query decomposition** — Groq LLM breaks complex questions into 2–3 keyword-dense sub-queries.
2. **Hybrid search** — A Postgres RPC (`hybrid_search`) fuses BM25 full-text ranking and HNSW vector search using Reciprocal Rank Fusion (RRF).
3. **Rerank** — A `cross-encoder/ms-marco-MiniLM-L-6-v2` CrossEncoder re-scores the retrieved chunks.
4. **Generate** — Groq streams the final answer, grounded strictly in the retrieved context.

### Key Decisions

- **Supabase + pgvector** — eliminates a separate vector store; hybrid search and metadata filtering happen in a single SQL query.
- **Transaction-mode PgBouncer (port 6543)** — Supabase's default pooler; prepared statements are disabled (`prepare_threshold=None`) to stay compatible.
- **`autocommit=True` during retrieval** — releases the backend connection back to the pool before the CPU-bound CrossEncoder runs, preventing connection pinning.
- **Context block per page** — LlamaParse is instructed to prepend a 2–3 sentence `Context:` summary to every page, so chunks that lack headings still carry document-level metadata into the embedding space.

---

## API Keys

| Service | Purpose | Get Key |
|---|---|---|
| Groq | LLM inference (query decomp + generation) | [console.groq.com/keys](https://console.groq.com/keys) |
| LlamaCloud | PDF → Markdown parsing | [cloud.llamaindex.ai/api-key](https://cloud.llamaindex.ai/api-key) |
| Supabase | Postgres + pgvector database | [supabase.com/dashboard](https://supabase.com/dashboard) → Settings → Database |
| Cohere | (optional, currently unused) | [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys) |
| OpenRouter | (optional, alternative LLM routing) | [openrouter.ai/keys](https://openrouter.ai/keys) |
