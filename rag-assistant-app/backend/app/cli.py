"""Command line tools.

    python -m app.cli ingest                       # build the vector store from the raw documents folder
    python -m app.cli ingest --raw-dir ./my_docs   # ... from another folder
"""
import argparse
import sys

from app.core.config import get_settings
from app.services.indexer import build_index
from app.utils.logging_config import setup_logging


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="RAG assistant maintenance commands")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="(re)build the vector store from a folder of PDF/TXT/MD files")
    ingest.add_argument("--raw-dir", default=None, help=f"documents folder (default: {settings.resolved_raw_dir})")
    ingest.add_argument("--store-dir", default=None, help=f"output folder (default: {settings.vector_store_dir})")
    ingest.add_argument("--embedding-model", default=None, help=f"default: {settings.embedding_model}")
    args = parser.parse_args(argv)

    setup_logging(settings.log_level, settings.log_format)
    from pathlib import Path

    raw_dir = Path(args.raw_dir) if args.raw_dir else settings.resolved_raw_dir
    store_dir = Path(args.store_dir) if args.store_dir else settings.vector_store_dir
    try:
        count = build_index(
            raw_dir, store_dir, args.embedding_model or settings.embedding_model,
            settings.chunk_size, settings.chunk_overlap,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Indexed {count} chunks into {store_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
