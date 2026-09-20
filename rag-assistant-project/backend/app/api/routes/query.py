from fastapi import APIRouter, HTTPException, Request

from app.schemas.query import QueryRequest, QueryResponse
from app.services.generation import LLMUnavailableError

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    retriever = getattr(request.app.state, "retriever", None)
    return {
        "status": "ok",
        "vector_store_loaded": retriever is not None,
        "chunks": retriever.chunk_count if retriever is not None and hasattr(retriever, "chunk_count") else None,
    }


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    retriever = request.app.state.retriever
    generator = request.app.state.generator
    chunks = retriever.retrieve(payload.question)
    try:
        answer, sources = generator.answer(payload.question, chunks)
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return QueryResponse(answer=answer, sources=sources)
