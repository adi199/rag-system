import os
import glob
import html
import re
import logging
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()

# Module-level logger setup
logger = logging.getLogger(__name__)


DATA_DIR = "data"
PDF_GLOB = f"{DATA_DIR}/*.pdf"

PARSING_INSTRUCTION = """
You are an expert financial data analyst extracting data for a semantic search index.

For EVERY page, output in this exact order:

---BEGIN_PAGE---

1. CONTEXT BLOCK (required, always first):
   Write 2-3 sentences summarising what this page is about.
   Always include: the correct company name (from Step 0), the presentation date,
   and the core topics or metrics on this page.
   If the page is a section divider with no data, say so explicitly.
   Format exactly:
   Context: <your 2-3 sentence summary here>

2. CONTENT (after the context block):

   A. HEADINGS: Preserve any slide or section title as a Markdown heading (## Heading).

   B. TABLES: Keep all tabular data as a valid Markdown table.
      - Always include a header row with column names.
      - If a table has multiple columns representing different entities (e.g. cities,
        years, or business units), label every column explicitly — never drop headers.
      - Present each dataset ONCE only. Do not repeat the same numbers as both a table
        AND a bullet list below it. Choose the table format.

   C. KPI BOXES & CALLOUTS: For every standalone metric, figure, or KPI callout,
      output it as a labeled key-value pair:
        Metric Name: Value
      Example:  Equity Market Capitalization: $60 Bn
      Do NOT leave any number without its label.

   D. CHARTS & GRAPHS — STRICT RULES:
      - Extract only explicitly labeled data point values with their category or year.
      - Do NOT extract axis tick marks, scale labels, or gridline values as data points.
        (e.g. if a Y-axis shows 0, 50, 100, 150, 200 — do not list those as data)
      - If a chart's actual values cannot be clearly read, output exactly:
        Chart present – data not extractable. Description: <one sentence on what the chart shows>
      - NEVER invent, estimate, or fabricate chart values. No placeholder variables
        like $X M or $Y M. If the number is not legible, use the line above instead.

   E. EMPTY / DIVIDER PAGES: If a page contains only a title and no data,
      output only the Context block and the heading. Do not add bullet points
      saying "No metrics available" — leave the content section empty.

---END_PAGE---
"""

# ──────────────────────────────────────────────
# LlamaParse initialisation
# ──────────────────────────────────────────────

parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    # Agentic tier: visually aware, understands KPI boxes, multi-column layouts
    result_type="markdown",
    # Inject a hard page boundary that matches the ---BEGIN_PAGE--- / ---END_PAGE---
    # markers in the instruction. run_pipeline.py splits on PAGE_NUMBER tags;
    # the BEGIN marker also ensures the Context block never bleeds into the previous page.
    page_separator="\n---END_PAGE---\n<!-- PAGE_NUMBER: {page_number} -->\n---BEGIN_PAGE---\n",
    parsing_instruction=PARSING_INSTRUCTION,
    # Verbose so we can see progress per document
    verbose=True,
)

# ──────────────────────────────────────────────
# Parsing loop
# ──────────────────────────────────────────────

def _postprocess(text: str, pdf_path: str) -> str:
    """
    Three fixes applied after LlamaParse returns:

    1. Decode HTML entities — fixes &#x26; → & and &#x27; → ' so BM25 keyword
       matching works on terms like "Q&A" and "BXP's".

    2. Collapse runs of blank lines — LlamaParse sometimes emits 3-4 blank lines
       between sections; normalise to a maximum of two.

    3. Company-name hallucination guard — warn loudly if a known-bad company name
       appears inside a Context block so it can be manually reviewed. We cannot
       auto-correct (we don't know the right name until runtime), but the warning
       surfaces the fault immediately.
    """
    # Fix 1: decode HTML entities
    text = html.unescape(text)

    # Fix 2: normalise excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Fix 3: warn on suspicious company names inside Context lines
    # Extend this list with any names that keep appearing incorrectly
    HALLUCINATED_NAMES = ["Fannie Mae", "Freddie Mac", "Sallie Mae"]
    context_lines = [l for l in text.splitlines() if l.startswith("Context:")]
    for line in context_lines:
        for bad_name in HALLUCINATED_NAMES:
            if bad_name.lower() in line.lower():
                logger.warning(
                    f"Possible company hallucination in {pdf_path}:\n"
                    f"         '{line[:120]}'\n"
                    f"         → Contains '{bad_name}'. Please review this page."
                )

    return text


def parse_pdf(pdf_path: str) -> None:
    """Parse a single PDF and write the enriched Markdown alongside it."""
    md_path = str(Path(pdf_path).with_suffix(".md"))

    logger.info(f"Parsing: {pdf_path} → Output: {md_path}")

    # load_data returns a list of Document objects (one per page)
    documents = parser.load_data(pdf_path)

    if not documents:
        logger.warning(f"No content returned for {pdf_path}")
        return

    # Join with a newline between documents so the last character of page N
    # never runs directly into the first character of page N+1.
    # The page_separator already embeds ---END_PAGE--- / PAGE_NUMBER / ---BEGIN_PAGE---,
    # so a single "\n" here is sufficient — it just prevents the rare case where
    # LlamaParse omits a trailing newline on a page.
    combined_md = "\n".join(doc.text for doc in documents)

    # Apply post-processing fixes
    combined_md = _postprocess(combined_md, pdf_path)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(combined_md)

    logger.info(f"Written {len(combined_md):,} chars → {md_path}")


def main():
    pdf_files = sorted(glob.glob(PDF_GLOB))

    if not pdf_files:
        logger.warning(f"No PDF files found in {DATA_DIR}/")
        return

    logger.info(f"Found {len(pdf_files)} PDF file(s) to parse.")

    from tqdm import tqdm
    pbar = tqdm(pdf_files, desc="Parsing PDFs")
    for pdf in pbar:
        pbar.set_description(f"Parsing {os.path.basename(pdf)}")
        try:
            parse_pdf(pdf)
        except Exception as e:
            logger.error(f"Failed to parse {pdf}: {e}", exc_info=True)

    logger.info("All done! Run `python run_pipeline.py` to re-index Supabase.")


if __name__ == "__main__":
    # Configure logging only when executed directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()