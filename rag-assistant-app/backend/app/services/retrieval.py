"""Hybrid retrieval (dense embeddings + BM25 fused with RRF) over a persisted Chroma store."""
import json
import logging
import threading
from collections import Counter
from dataclasses import dataclass

import chromadb

from app.core.config import Settings
from app.services.bm25 import BM25Index, fuse_rankings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float  # cosine similarity of the dense embedding, higher is better
    chunk_id: str = ""

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
        self.chunk_size = int(self.config.get("chunk_size", 800))
        self.chunk_overlap = int(self.config.get("chunk_overlap", 100))

        # Imported here so that importing the app (e.g. in tests) stays fast.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s", self.config["embedding_model"])
        self.embedder = SentenceTransformer(self.config["embedding_model"])

        client = chromadb.PersistentClient(path=str(store_dir / "chroma_db"))
        self.collection = client.get_collection(self.config["collection_name"])
        self.top_k = settings.top_k
        self.min_similarity = settings.min_similarity
        self._lock = threading.Lock()
        self._bm25: BM25Index | None = None
        self._rebuild_index()
        logger.info("Vector store loaded: %d chunks", self.collection.count())

    # ------------------------------------------------------------------ search
    @property
    def chunk_count(self) -> int:
        return self.collection.count()

    def _rebuild_index(self) -> None:
        data = self.collection.get(include=["documents"])
        self._bm25 = BM25Index(list(data["ids"]), list(data["documents"]))

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Dense candidates re-ranked with BM25 via Reciprocal Rank Fusion."""
        top_k = top_k or self.top_k
        total = self.collection.count()
        if total == 0:
            return []
        n_candidates = max(1, min(max(top_k * 3, 12), total))

        embedding = self.embedder.encode([question], normalize_embeddings=True).tolist()
        result = self.collection.query(
            query_embeddings=embedding,
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )
        ids = list(result["ids"][0])
        candidates: dict[str, RetrievedChunk] = {}
        for chunk_id, text, meta, dist in zip(
            ids, result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            # Chroma returns squared L2 distance; for unit vectors cos = 1 - d/2.
            candidates[chunk_id] = RetrievedChunk(
                text=text, source=meta["source"], page=int(meta["page"]),
                score=1.0 - dist / 2.0, chunk_id=chunk_id,
            )

        lexical = self._bm25.scores(question) if self._bm25 else {}
        lexical_rank = sorted((i for i in ids if lexical.get(i, 0.0) > 0), key=lambda i: lexical[i], reverse=True)
        fused = fuse_rankings([ids, lexical_rank])
        ordered = sorted(candidates.values(), key=lambda c: fused[c.chunk_id], reverse=True)

        relevant = [c for c in ordered if c.score >= self.min_similarity]
        logger.info("Retrieved %d/%d chunks above threshold %.2f", len(relevant[:top_k]), len(ordered), self.min_similarity)
        return relevant[:top_k]

    # ------------------------------------------------------- knowledge base CRUD
    def list_documents(self) -> list[dict]:
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        counts = Counter(m["source"] for m in metas)
        return [{"name": name, "chunks": n} for name, n in sorted(counts.items())]

    def add_chunks(self, chunks: list[dict]) -> int:
        """Index chunks, replacing any existing chunks of the same source document."""
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts, batch_size=32, normalize_embeddings=True).tolist()
        with self._lock:
            for source in {c["source"] for c in chunks}:
                self._delete_source(source)
            self.collection.add(
                ids=[c["id"] for c in chunks],
                embeddings=embeddings,
                documents=texts,
                metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
            )
            self._rebuild_index()
        logger.info("Indexed %d chunks from %s", len(chunks), sorted({c["source"] for c in chunks}))
        return len(chunks)

    def delete_document(self, name: str) -> int:
        with self._lock:
            deleted = self._delete_source(name)
            if deleted:
                self._rebuild_index()
        return deleted

    def _delete_source(self, name: str) -> int:
        ids = self.collection.get(where={"source": name}, include=["metadatas"])["ids"]
        if ids:
            self.collection.delete(ids=list(ids))
        return len(ids)
