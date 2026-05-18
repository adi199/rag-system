# RAG System

A document question-answering system powered by **Supabase + pgvector**, **LlamaParse**, and **Groq**.

Drop PDFs into `data/`, index them, and ask questions in a chat UI.

<!-- MEDIA: screenshot of the Streamlit chat interface -->

---

## Quickstart

**Prerequisites:** [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager

Install uv if you don't have it:

### macOS / Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 1. Clone and enter the project
```bash
git clone https://github.com/adi199/rag-system && cd rag-system
```

### 2. Copy and fill in your API keys
```bash
cp .env.example .env
```

### 3. Install dependencies
```bash
uv sync
```

### 4. Drop your PDFs into data/, then run the full setup
```bash
uv run python manage.py setup
```

### 5. Launch the app
```bash
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

### Adding new documents to an existing deployment
```bash
cp new_docs/*.pdf data/
```

### Re-ingest (parse + embed + store) only the new PDFs
```bash
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
| Groq | LLM inference (query decomp + generation) | [console.groq.com](https://console.groq.com) |
| LlamaCloud | PDF → Markdown parsing | [cloud.llamaindex.ai](https://cloud.llamaindex.ai) |
| Supabase | Postgres + pgvector database | [supabase.com](https://supabase.com) |
