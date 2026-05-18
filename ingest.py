import os
import glob
import html
import re
import json
import psycopg
import logging

from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
import semchunk
from db import DB_KWARGS

# Module-level logger setup
logger = logging.getLogger(__name__)

# ── Embedding model ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ── Chunker ────────────────────────────────────────────────────────────────────
chunker = semchunk.chunkerify("sentence-transformers/all-MiniLM-L6-v2", chunk_size=512)

# ── Regex constants ────────────────────────────────────────────────────────────

# New format (updated parse_docs.py): separator carries the NEXT page's number
_SEP_WITH_PAGENUM = re.compile(
    r'---END_PAGE---\s*<!-- PAGE_NUMBER: (\d+) -->\s*---BEGIN_PAGE---'
)
# New format (test / legacy new): BEGIN/END markers but no page numbers
_SEP_NO_PAGENUM = re.compile(r'---BEGIN_PAGE---')
# Old format: only page number comments, no BEGIN/END markers
_SEP_OLD = re.compile(r'<!-- PAGE_NUMBER: (\d+) -->\n?')

# LlamaParse structural label lines injected by the instruction template —
# these are meta-commentary that should not end up in embeddings.
# Matches lines like:
#   "1. CONTEXT BLOCK (required, always first):"
#   "2. CONTENT (after the context block):"
#   "A. HEADINGS:"  "B. TABLES:"  "C. KPI BOXES & CALLOUTS:"
#   "D. CHARTS & GRAPHS — STRICT RULES:"  "E. EMPTY / DIVIDER PAGES:"
#   "KPI Box:"
_NOISE_LINE = re.compile(
    r'^(\d+\.\s+(CONTEXT BLOCK|CONTENT)\b.*'
    r'|[A-E]\.\s+(HEADINGS|TABLES|KPI BOXES|CHARTS|EMPTY)\b.*'
    r'|KPI Box:\s*)$',
    re.IGNORECASE
)

# "Metric Name: " prefix added by LlamaParse to KPI lines — strip it so
# "Metric Name: Equity Market Cap: $60 Bn" becomes "Equity Market Cap: $60 Bn"
_METRIC_NAME_PREFIX = re.compile(r'^Metric Name:\s*', re.MULTILINE)

# Minimum body length (chars) after stripping noise and the Context line.
# Pages below this threshold are section dividers with no retrievable content
# and are skipped entirely rather than creating near-empty chunks.
_DIVIDER_THRESHOLD = 80


# ── Per-page cleaning ──────────────────────────────────────────────────────────

def _clean_page_text(raw: str) -> str:
    """
    Given the raw text of one page (between BEGIN/END markers), return clean
    text ready for chunking:
      1. Decode HTML entities  (&amp; → &, &#x26; → &, etc.)
      2. Strip structural label lines injected by the parsing instruction
      3. Strip "Metric Name: " prefixes from KPI lines
      4. Collapse excess blank lines to at most two
    """
    text = html.unescape(raw)
    # Remove noise label lines
    lines = [l for l in text.splitlines() if not _NOISE_LINE.match(l.strip())]
    text = "\n".join(lines)
    # Normalise "Metric Name: X: Y" → "X: Y"
    text = _METRIC_NAME_PREFIX.sub("", text)
    # Collapse 3+ blank lines → 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _extract_context(page_text: str) -> str:
    """
    Extract the Context: ... line from the page text.
    Returns the full line including the 'Context:' prefix, or '' if absent.
    """
    for line in page_text.splitlines():
        if line.strip().lower().startswith("context:"):
            return line.strip()
    return ""


def _body_length(page_text: str) -> int:
    """
    Return the character length of page content AFTER the Context line.
    Used to detect section-divider pages with no real data.
    """
    in_body = False
    body_chars = 0
    for line in page_text.splitlines():
        if in_body:
            body_chars += len(line.strip())
        if line.strip().lower().startswith("context:"):
            in_body = True
    return body_chars


# ── Format detection & splitting ──────────────────────────────────────────────

def _split_into_pages(markdown_text: str) -> list[tuple[int, str]]:
    """
    Detect the markdown format and split into (page_number, page_text) tuples.

    Three formats are handled, in priority order:
      1. New format WITH page numbers:
           [page 1 content]
           ---END_PAGE---
           <!-- PAGE_NUMBER: 2 -->
           ---BEGIN_PAGE---
           [page 2 content]
           ...
         The page number in the separator is the number of the FOLLOWING page,
         so page 1 gets number 1 (implicit), page 2 gets number 2, etc.

      2. New format WITHOUT page numbers (BEGIN/END markers only):
           ---BEGIN_PAGE---
           [page content]
           ---END_PAGE---
           ---BEGIN_PAGE---
           [page content]
           ...
         Page numbers are assigned sequentially starting at 1.

      3. Old format (<!-- PAGE_NUMBER: N --> only, no BEGIN/END):
           <!-- PAGE_NUMBER: 3 -->
           [page content]
           ...
         Page numbers come from the comment tags.
    """
    # Format 1: new format with page numbers embedded in separator
    if _SEP_WITH_PAGENUM.search(markdown_text):
        parts = _SEP_WITH_PAGENUM.split(markdown_text)
        # parts = [before_page2, "2", page2_text, "3", page3_text, ...]
        # The very first segment (before the first separator) is page 1.
        results = []
        first_page = parts[0].strip()
        # Strip the opening ---BEGIN_PAGE--- if present
        first_page = re.sub(r'^---BEGIN_PAGE---\s*', '', first_page).strip()
        # Strip trailing ---END_PAGE--- from page body
        first_page = re.split(r'---END_PAGE---', first_page)[0].strip()
        if first_page:
            results.append((1, first_page))
        for i in range(1, len(parts), 2):
            page_num = int(parts[i])
            page_body = parts[i + 1] if i + 1 < len(parts) else ""
            page_body = re.split(r'---END_PAGE---', page_body)[0].strip()
            if page_body:
                results.append((page_num, page_body))
        return results

    # Format 2: BEGIN/END markers without page numbers
    if _SEP_NO_PAGENUM.search(markdown_text):
        raw_pages = _SEP_NO_PAGENUM.split(markdown_text)
        results = []
        for idx, raw in enumerate(raw_pages, start=1):
            body = re.split(r'---END_PAGE---', raw)[0].strip()
            if body:
                results.append((idx, body))
        return results

    # Format 3: old <!-- PAGE_NUMBER: N --> format
    if _SEP_OLD.search(markdown_text):
        parts = _SEP_OLD.split(markdown_text)
        results = []
        for i in range(1, len(parts), 2):
            page_num = int(parts[i])
            page_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if page_text:
                results.append((page_num, page_text))
        return results

    # Fallback: no recognised markers — treat whole file as one page
    return [(1, markdown_text.strip())] if markdown_text.strip() else []


# ── Main chunking logic ────────────────────────────────────────────────────────

def section_aware_semantic_chunking(markdown_text: str, source_name: str) -> dict:
    docs, metadatas, ids = [], [], []
    chunk_counter = 0

    pages = _split_into_pages(markdown_text)

    if not pages:
        # Last-resort fallback: section-level splitting (old .md files, no markers)
        sections = re.split(r'(?=\n#+ )', markdown_text)
        for section in sections:
            section_text = section.strip()
            if not section_text:
                continue
            for chunk in chunker(section_text):
                docs.append(f"[Document: {source_name}, Page: N/A]\n{chunk}")
                metadatas.append({"source": source_name, "page": "N/A"})
                ids.append(f"{source_name}-chunk{chunk_counter}")
                chunk_counter += 1
        return {"documents": docs, "metadatas": metadatas, "ids": ids}

    for page_num, raw_page_text in pages:
        # Step 1: clean noise labels, entities, metric prefixes
        page_text = _clean_page_text(raw_page_text)

        # Step 2: skip section-divider pages (no useful body content)
        if _body_length(page_text) < _DIVIDER_THRESHOLD:
            continue

        # Step 3: extract the Context block to prepend to every chunk
        context_block = _extract_context(page_text)
        context_prefix = (context_block + "\n---\n") if context_block else ""

        # Step 4: semantic chunking of the full cleaned page text
        chunks = chunker(page_text)

        for chunk in chunks:
            # Prepend context unless the chunk already opens with it
            # (the first chunk of a page often includes the Context line itself)
            if context_prefix and context_block[:60] not in chunk:
                enriched = (
                    f"[Document: {source_name}, Page: {page_num}]\n"
                    f"{context_prefix}{chunk}"
                )
            else:
                enriched = f"[Document: {source_name}, Page: {page_num}]\n{chunk}"

            docs.append(enriched)
            metadatas.append({"source": source_name, "page": int(page_num)})
            ids.append(f"{source_name}-chunk{chunk_counter}")
            chunk_counter += 1

    return {"documents": docs, "metadatas": metadatas, "ids": ids}


# ── Storage & orchestration ────────────────────────────────────────────────────

def store_in_postgres(chunk_data: dict):
    """Embed each chunk and batch-insert into the Supabase documents table."""
    docs = chunk_data["documents"]
    metadatas = chunk_data["metadatas"]
    ids = chunk_data["ids"]

    if not docs:
        return

    # Generate embeddings for all chunks in one batch call
    embeddings = EMBEDDING_MODEL.encode(docs, show_progress_bar=True, normalize_embeddings=True)

    rows = [
        (
            ids[i],             # chunk_id
            docs[i],            # content
            json.dumps(metadatas[i]),  # metadata (JSONB)
            embeddings[i].tolist(),    # embedding
        )
        for i in range(len(docs))
    ]

    with psycopg.connect(**DB_KWARGS) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    """
                    INSERT INTO documents (chunk_id, content, metadata, embedding)
                    VALUES (%s, %s, %s::jsonb, %s)
                    """,
                    row,
                )
        conn.commit()

    logger.info(f"Stored {len(docs)} chunks in Supabase.")


def process_file(md_path: str):
    logger.info(f"Processing {md_path}...")
    base_name = os.path.basename(md_path).replace(".md", ".pdf")

    with open(md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    chunk_data = section_aware_semantic_chunking(markdown_text, base_name)
    store_in_postgres(chunk_data)
    logger.info(f"Finished processing {md_path}.")


def main():
    md_files = glob.glob("data/*.md")
    if not md_files:
        logger.warning("No markdown files found to process.")
        return

    logger.info(f"Found {len(md_files)} markdown file(s) to process.")

    from tqdm import tqdm
    pbar = tqdm(md_files, desc="Indexing Documents")
    for md in pbar:
        pbar.set_description(f"Indexing {os.path.basename(md)}")
        process_file(md)
    logger.info("All processing finished.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()