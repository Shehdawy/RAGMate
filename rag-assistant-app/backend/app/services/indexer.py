"""Build the vector store from a folder of documents (used by the CLI and AUTO_INGEST)."""
import json
import logging
from pathlib import Path

from app.services.ingestion import IngestionError, build_chunks, extract_pages

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
COLLECTION_NAME = "documents"


def collect_chunks(raw_dir: Path, chunk_size: int = 800, overlap: int = 100) -> tuple[list[dict], list[str]]:
    """Chunk every supported file in raw_dir. Returns (chunks, skipped file messages)."""
    chunks: list[dict] = []
    skipped: list[str] = []
    for path in sorted(Path(raw_dir).iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            pages = extract_pages(path.name, path.read_bytes())
        except IngestionError as exc:
            skipped.append(f"{path.name}: {exc}")
            continue
        chunks.extend(build_chunks(path.name, pages, chunk_size, overlap))
    return chunks, skipped


def store_config(embedding_model: str, chunk_size: int, overlap: int, chunks: list[dict]) -> dict:
    return {
        "embedding_model": embedding_model,
        "collection_name": COLLECTION_NAME,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "similarity": "squared L2 on normalized embeddings (cosine = 1 - d/2)",
        "num_chunks": len(chunks),
        "source_files": sorted({c["source"] for c in chunks}),
        "built_by": "app.cli ingest",
    }


def build_index(raw_dir: Path, store_dir: Path, embedding_model: str, chunk_size: int = 800, overlap: int = 100) -> int:
    """Embed all documents and write a persistent Chroma store plus config.json. Returns the chunk count."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    chunks, skipped = collect_chunks(raw_dir, chunk_size, overlap)
    for message in skipped:
        logger.warning("Skipped %s", message)
    if not chunks:
        raise ValueError(f"No indexable documents (PDF, TXT, MD) found in {raw_dir}")

    logger.info("Embedding %d chunks with %s", len(chunks), embedding_model)
    embedder = SentenceTransformer(embedding_model)
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False).tolist()

    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(store_dir / "chroma_db"))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection did not exist yet
    collection = client.create_collection(COLLECTION_NAME)
    for start in range(0, len(chunks), 500):
        batch = slice(start, start + 500)
        collection.add(
            ids=[c["id"] for c in chunks[batch]],
            embeddings=embeddings[batch],
            documents=texts[batch],
            metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks[batch]],
        )
    (store_dir / "config.json").write_text(
        json.dumps(store_config(embedding_model, chunk_size, overlap, chunks), indent=2), encoding="utf-8"
    )
    logger.info("Vector store written to %s (%d chunks)", store_dir, len(chunks))
    return len(chunks)


def ensure_vector_store(settings) -> None:
    """On first start, build the store from raw documents when AUTO_INGEST is enabled."""
    if (settings.vector_store_dir / "config.json").exists():
        return
    if not settings.auto_ingest:
        return  # Retriever raises a clear error telling the user to build the store
    raw_dir = settings.resolved_raw_dir
    logger.info("No vector store found: building one from %s", raw_dir)
    build_index(raw_dir, settings.vector_store_dir, settings.embedding_model, settings.chunk_size, settings.chunk_overlap)
