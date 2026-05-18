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

### High-Level Architecture

```
PDFs (data/)
  └─ parse.py (LlamaParse API)
       └─ Markdown files (data/*.md)
            └─ ingest.py (semchunk + all-MiniLM-L6-v2)
                 └─ Supabase PostgreSQL (pgvector)
                      └─ app.py (Streamlit)
                           ├─ Groq: query decomposition  (llama-4-scout-17b)
                           ├─ hybrid_search() RPC        (BM25 + HNSW via RRF)
                           ├─ CrossEncoder rerank        (ms-marco-MiniLM-L-6-v2)
                           └─ Groq: answer generation    (llama-3.3-70b-versatile)
```

### Database

**Supabase PostgreSQL** (not a data warehouse) with the `pgvector` extension. A single `documents` table stores all chunks:

| Column | Type | Description |
|---|---|---|
| `id` | `BIGINT` (identity) | Primary key |
| `chunk_id` | `TEXT` | `{filename}-chunk{N}` |
| `content` | `TEXT` | Full enriched chunk text |
| `metadata` | `JSONB` | `{"source": filename, "page": N}` |
| `embedding` | `vector(384)` | Normalized `all-MiniLM-L6-v2` embedding |
| `fts` | `tsvector` (generated) | Auto-updated BM25 index over `content` |

Indexes: **GIN** on `fts`, **HNSW** (`vector_ip_ops`) on `embedding`.

The database is accessed via Supabase's PgBouncer pooler in **transaction mode** (port 6543). Prepared statements are disabled (`prepare_threshold=None`) for compatibility. During retrieval, `autocommit=True` ensures the connection is released back to the pool before the CPU-bound CrossEncoder runs.

### Chunking Strategy

Each PDF page is processed independently:

1. LlamaParse emits a `Context:` line (2–3 sentence summary) at the top of every page — carrying company name, date, and key topics.
2. The full page text is cleaned: HTML entities decoded, LlamaParse structural label lines stripped, `Metric Name:` prefixes removed, excess blank lines collapsed.
3. Pages with fewer than 80 characters of body content after the `Context:` line are dropped (section dividers).
4. `semchunk` splits the cleaned page text semantically using the `all-MiniLM-L6-v2` tokenizer at a **512-token chunk size**.
5. Every chunk gets a `[Document: filename, Page: N]` header prepended. If the chunk doesn't already open with the `Context:` line, that line is also prepended — ensuring every chunk carries page-level context into the embedding space.

Chunk IDs follow the pattern `{source_filename}-chunk{N}` and are stored alongside `source` and `page` in the `metadata` JSONB column.

### Retrieval Approach

1. **Query decomposition** — `llama-4-scout-17b` (Groq, temperature=0) rewrites the user question into 2–3 keyword-dense sub-queries. Each is searched independently.
2. **Hybrid search** — Each sub-query hits a Postgres RPC (`hybrid_search`) that fuses:
   - BM25 full-text ranking (`websearch_to_tsquery`)
   - HNSW approximate nearest-neighbour search (inner product)
   - Combined via **Reciprocal Rank Fusion** (`rrf_k=50`, equal weights)
   - Returns up to 20 candidates per sub-query.
3. **CrossEncoder rerank** — `ms-marco-MiniLM-L-6-v2` scores each (query, chunk) pair; top 5 per sub-query are kept.
4. **Deduplication** — chunks seen across sub-queries are deduplicated by content before being passed to the LLM.
5. **Generation** — `llama-3.3-70b-versatile` (Groq, temperature=0, streaming) generates the answer strictly from the retrieved context with mandatory `[Source: chunk_id]` citations.

### How Specific Cases Are Handled

**Versioning**
There is no explicit version field in the schema. Documents are distinguished by filename — the `source` metadata field contains the original PDF filename, and the `chunk_id` encodes it as a prefix. The LLM system prompt explicitly instructs the model to treat documents as potentially representing different versions or time periods, never blend values silently, and report each separately with its source citation.

**Conflicting information**
Handled entirely at the prompt level. The system prompt instructs the LLM: *"If documents contain contradictory information, surface the conflict explicitly."* There is no retrieval-level conflict detection.

**Charts and tables**
LlamaParse is given strict per-category instructions:
- **Tables** → emitted as valid Markdown tables with explicit column headers; each dataset presented once only.
- **KPI boxes / callouts** → emitted as `Metric Name: Value` key-value pairs; no number is left unlabeled.
- **Charts** → only explicitly labeled data points are extracted. If values cannot be clearly read, the output is `Chart present – data not extractable. Description: ...`. Fabrication is explicitly forbidden in the parsing instruction.

The LLM system prompt mirrors this: it forbids outputting raw HTML or re-encoding entities, and requires all tabular data to use standard Markdown table syntax.

---

### Known Limitations

- **No upsert on re-ingestion.** Running `ingest` on an already-indexed document appends duplicate chunks rather than replacing them. `setup` avoids this by dropping the table first, but `reindex` does not.
- **Versioning is filename-based.** Two files with the same name but different content cannot be distinguished in the index.
- **Section-divider threshold is a fixed constant** (80 chars). Pages with sparse but meaningful content may be incorrectly dropped.
- **LlamaParse hallucination guard is warn-only.** A hardcoded list of known-bad company names triggers a log warning but no automatic correction.
- **No evaluation framework.** Retrieval quality is assessed manually; there are no automated recall/precision metrics.
- **Single-tenant index.** All documents share one `documents` table with no namespace isolation.

### What Would Be Improved With More Time

- **Upsert logic** — `INSERT ... ON CONFLICT (chunk_id) DO UPDATE` to make re-ingestion idempotent.
- **Explicit version/date metadata field** in the schema, populated from filename or document frontmatter, enabling filtered retrieval by version.
- **Retrieval evaluation pipeline** — a small labelled Q&A set to measure recall@k and track regressions when chunking or search parameters change.
- **Async ingestion with progress streaming** to the Streamlit UI.
- **Namespace / collection isolation** per document set, so different corpora can coexist in one database without cross-contamination.

---

## API Keys

| Service | Purpose | Get Key |
|---|---|---|
| Groq | LLM inference (query decomp + generation) | [console.groq.com](https://console.groq.com) |
| LlamaCloud | PDF → Markdown parsing | [cloud.llamaindex.ai](https://cloud.llamaindex.ai) |
| Supabase | Postgres + pgvector database | [supabase.com](https://supabase.com) |
