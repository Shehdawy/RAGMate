"""Loads the persisted vector store once and retrieves relevant chunks."""
import json
import logging
from dataclasses import dataclass

import chromadb

from app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float  # cosine similarity, higher is better

    @property
    def label(self) -> str:
        return f"{self.source} (page {self.page})"


class Retriever:
    def __init__(self, settings: Settings):
        store_dir = settings.vector_store_dir
        config_path = store_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_path} not found. Run notebooks/rag_pipeline.ipynb first "
                "to build the vector store."
            )
        self.config = json.loads(config_path.read_text(encoding="utf-8"))

        # Imported here so that importing the app (e.g. in tests) stays fast.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s", self.config["embedding_model"])
        self.embedder = SentenceTransformer(self.config["embedding_model"])

        client = chromadb.PersistentClient(path=str(store_dir / "chroma_db"))
        self.collection = client.get_collection(self.config["collection_name"])
        self.top_k = settings.top_k
        self.min_similarity = settings.min_similarity
        logger.info("Vector store loaded: %d chunks", self.collection.count())

    @property
    def chunk_count(self) -> int:
        return self.collection.count()

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        embedding = self.embedder.encode([question], normalize_embeddings=True).tolist()
        result = self.collection.query(
            query_embeddings=embedding,
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks: list[RetrievedChunk] = []
        for text, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            # Chroma returns squared L2 distance; for unit vectors cos = 1 - d/2.
            score = 1.0 - dist / 2.0
            if score >= self.min_similarity:
                chunks.append(
                    RetrievedChunk(text=text, source=meta["source"], page=int(meta["page"]), score=score)
                )
        logger.info("Retrieved %d chunks above threshold %.2f", len(chunks), self.min_similarity)
        return chunks
