"""
manage.py
---------
Unified CLI for the RAG system. Run `uv run python manage.py --help` to see all commands.
"""

import argparse
import logging
import sys


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def cmd_init_db(_args):
    """Enable pgvector, create the documents table, indexes, and hybrid_search() RPC."""
    import init_db
    init_db.main()


def cmd_parse(_args):
    """Parse all PDFs in data/ into Markdown files via LlamaParse."""
    import parse
    parse.main()


def cmd_ingest(_args):
    """Chunk, embed, and store all Markdown files in data/ into Supabase."""
    import ingest
    ingest.main()


def cmd_setup(args):
    """Full first-time setup: init_db → parse → ingest."""
    logging.getLogger(__name__).info("=== Step 1/3: Initialising database ===")
    cmd_init_db(args)
    logging.getLogger(__name__).info("=== Step 2/3: Parsing PDFs ===")
    cmd_parse(args)
    logging.getLogger(__name__).info("=== Step 3/3: Ingesting documents ===")
    cmd_ingest(args)
    logging.getLogger(__name__).info("=== Setup complete. Run: uv run streamlit run app.py ===")


def cmd_reindex(args):
    """Re-parse and re-ingest without touching the database schema. Use when adding new PDFs."""
    logging.getLogger(__name__).info("=== Step 1/2: Parsing PDFs ===")
    cmd_parse(args)
    logging.getLogger(__name__).info("=== Step 2/2: Ingesting documents ===")
    cmd_ingest(args)
    logging.getLogger(__name__).info("=== Reindex complete. ===")


def main():
    parser = argparse.ArgumentParser(
        prog="manage.py",
        description="RAG system management CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup",    help="Full first-time setup: init-db -> parse -> ingest")
    sub.add_parser("reindex",  help="Re-parse + re-ingest only (no DB schema changes). Use when adding new PDFs.")
    sub.add_parser("init-db",  help="Create DB schema, indexes, and hybrid_search() RPC function")
    sub.add_parser("parse",    help="Parse PDFs in data/ -> Markdown via LlamaParse")
    sub.add_parser("ingest",   help="Chunk, embed, and store Markdown files into Supabase")

    args = parser.parse_args()

    _setup_logging()

    dispatch = {
        "setup":   cmd_setup,
        "reindex": cmd_reindex,
        "init-db": cmd_init_db,
        "parse":   cmd_parse,
        "ingest":  cmd_ingest,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
