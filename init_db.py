"""
init_db.py
----------
Run once to prepare the Supabase Postgres database for the RAG system.

Steps:
  1. Enable the pgvector extension.
  2. Drop and re-create the 'documents' table.
  3. Create GIN (fts) and HNSW (embedding) indexes.
  4. Deploy the hybrid_search() Postgres RPC function.
"""

import psycopg
import logging
from db import DB_KWARGS

# Module-level logger setup
logger = logging.getLogger(__name__)

# PgBouncer (transaction mode) doesn't allow multiple commands in one execute().
# Each statement must be executed individually.
_SCHEMA_STMTS = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    "DROP TABLE IF EXISTS documents CASCADE",
    """
    CREATE TABLE documents (
        id       BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        chunk_id TEXT NOT NULL,
        content  TEXT NOT NULL,
        metadata JSONB NOT NULL DEFAULT '{}',
        embedding vector(384) NOT NULL,
        fts      tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
    )
    """,
    "CREATE INDEX IF NOT EXISTS documents_fts_idx ON documents USING gin(fts)",
    "CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING hnsw (embedding vector_ip_ops)",
]

# The hybrid_search function body is a single statement (no semicolons at top level).
_HYBRID_SEARCH_FN = """
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text      text,
    query_embedding vector(384),
    match_count     int,
    full_text_weight float = 1,
    semantic_weight  float = 1,
    rrf_k            int   = 50
)
RETURNS SETOF documents
LANGUAGE sql
AS $$
WITH full_text AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(fts, websearch_to_tsquery(query_text)) DESC
        ) AS rank_ix
    FROM documents
    WHERE fts @@ websearch_to_tsquery(query_text)
    ORDER BY rank_ix
    LIMIT LEAST(match_count, 30) * 2
),
semantic AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            ORDER BY embedding <#> query_embedding
        ) AS rank_ix
    FROM documents
    ORDER BY rank_ix
    LIMIT LEAST(match_count, 30) * 2
)
SELECT documents.*
FROM full_text
FULL OUTER JOIN semantic ON full_text.id = semantic.id
JOIN documents ON COALESCE(full_text.id, semantic.id) = documents.id
ORDER BY
    COALESCE(1.0 / (rrf_k + full_text.rank_ix),  0.0) * full_text_weight +
    COALESCE(1.0 / (rrf_k + semantic.rank_ix), 0.0) * semantic_weight
    DESC
LIMIT LEAST(match_count, 30);
$$
"""

def main():
    logger.info(f"Connecting to {DB_KWARGS['host']}:{DB_KWARGS['port']}/{DB_KWARGS['dbname']}...")
    try:
        with psycopg.connect(**DB_KWARGS) as conn:
            with conn.cursor() as cur:
                logger.info("Steps 1-3: Enabling extension, creating table and indexes...")
                for stmt in _SCHEMA_STMTS:
                    cur.execute(stmt)
                logger.info("Table 'documents' successfully created with fts and embedding columns.")

                logger.info("Step 4: Deploying hybrid_search() function...")
                cur.execute(_HYBRID_SEARCH_FN)
                logger.info("hybrid_search() RPC function successfully deployed.")

            conn.commit()
        logger.info("Database setup complete. Ready to ingest documents.")
    except Exception as e:
        logger.exception("Failed to complete database setup")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
