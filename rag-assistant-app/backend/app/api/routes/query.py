import json
import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from app.core.metrics import metrics
from app.core.security import rate_limit, verify_api_key
from app.schemas.query import Citation, QueryRequest, QueryResponse
from app.services.generation import LLMUnavailableError, pick_sources, select_cited
from app.services.language import is_refusal
from app.services.retrieval import RetrievedChunk

router = APIRouter()
PROTECTED = [Depends(verify_api_key), Depends(rate_limit)]  # API key + rate limit


def _snippet(text: str, limit: int = 280) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def build_citations(pairs: list[tuple[int, RetrievedChunk]]) -> list[Citation]:
    return [
        Citation(ref=ref, source=c.source, page=c.page, score=round(c.score, 3), snippet=_snippet(c.text))
        for ref, c in pairs
    ]


def _event(payload: dict) -> str:
    return json.dumps(payload) + "\n"


@router.get("/health", tags=["system"])
def health(request: Request) -> dict:
    retriever = getattr(request.app.state, "retriever", None)
    generator = getattr(request.app.state, "generator", None)
    info = {
        "status": "ok",
        "vector_store_loaded": retriever is not None,
        "chunks": None,
        "documents": None,
        "model": getattr(generator, "model", None),
    }
    if retriever is not None:
        info["chunks"] = getattr(retriever, "chunk_count", None)
        list_documents = getattr(retriever, "list_documents", None)
        if callable(list_documents):
            info["documents"] = len(list_documents())
    return info


@router.get("/ready", tags=["system"])
def ready(request: Request) -> JSONResponse:
    """Readiness probe: the vector store is loaded and the LLM provider answers."""
    retriever = getattr(request.app.state, "retriever", None)
    generator = getattr(request.app.state, "generator", None)
    checks = {
        "vector_store": retriever is not None and bool(getattr(retriever, "chunk_count", 0)),
        "llm": bool(generator is not None and generator.check_connection()),
    }
    is_ready = all(checks.values())
    return JSONResponse(status_code=200 if is_ready else 503, content={"ready": is_ready, "checks": checks})


@router.get("/metrics", tags=["system"], include_in_schema=False)
def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


@router.post("/query", response_model=QueryResponse, tags=["query"], dependencies=PROTECTED)
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    started = time.perf_counter()
    retriever = request.app.state.retriever
    generator = request.app.state.generator
    chunks = retriever.retrieve(payload.question, payload.top_k)
    try:
        answer, sources = generator.answer(payload.question, chunks)
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    metrics.inc("rag_queries_total")
    if is_refusal(answer):
        metrics.inc("rag_refusals_total")
    return QueryResponse(
        answer=answer,
        sources=sources,
        citations=build_citations(select_cited(answer, chunks)),
        latency_ms=round((time.perf_counter() - started) * 1000),
        model=getattr(generator, "model", None),
    )


@router.post("/query/stream", tags=["query"], dependencies=PROTECTED)
def query_stream(payload: QueryRequest, request: Request) -> StreamingResponse:
    """Same as /query but streams newline-delimited JSON events: meta, token..., done | error."""
    started = time.perf_counter()
    retriever = request.app.state.retriever
    generator = request.app.state.generator
    # Retrieve before streaming so a failure here returns a proper HTTP error.
    chunks = retriever.retrieve(payload.question, payload.top_k)

    def events() -> Iterator[str]:
        yield _event({"type": "meta", "retrieved": len(chunks)})
        parts: list[str] = []
        try:
            for token in generator.stream(payload.question, chunks):
                parts.append(token)
                yield _event({"type": "token", "text": token})
        except LLMUnavailableError as exc:
            yield _event({"type": "error", "detail": str(exc)})
            return
        answer = "".join(parts).strip()
        metrics.inc("rag_queries_total")
        if is_refusal(answer):
            metrics.inc("rag_refusals_total")
        yield _event({
            "type": "done",
            "sources": pick_sources(answer, chunks),
            "citations": [c.model_dump() for c in build_citations(select_cited(answer, chunks))],
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "model": getattr(generator, "model", None),
        })

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
