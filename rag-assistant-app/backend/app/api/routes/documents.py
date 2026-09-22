from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.core.config import get_settings
from app.core.security import rate_limit, verify_api_key
from app.schemas.documents import DeleteResponse, DocumentInfo, UploadResponse
from app.services.ingestion import (
    ALLOWED_SUFFIXES,
    IngestionError,
    build_chunks,
    extract_pages,
    safe_filename,
)

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(verify_api_key)])


@router.get("", response_model=list[DocumentInfo])
def list_documents(request: Request) -> list[dict]:
    return request.app.state.retriever.list_documents()


@router.post("", response_model=UploadResponse, dependencies=[Depends(rate_limit)])
def upload_document(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    """Index a PDF/TXT/MD file. Re-uploading a file with the same name replaces it."""
    retriever = request.app.state.retriever
    name = safe_filename(file.filename)
    if not name:
        raise HTTPException(status_code=400, detail="This file has no name.")
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="This file type isn't supported. Please upload a PDF, TXT or MD file.")

    max_mb = get_settings().max_upload_mb
    data = file.file.read(max_mb * 1024 * 1024 + 1)
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"This file is too large (maximum {max_mb} MB).")

    try:
        pages = extract_pages(name, data)
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    chunks = build_chunks(name, pages, retriever.chunk_size, retriever.chunk_overlap)
    retriever.add_chunks(chunks)
    return UploadResponse(document=name, pages=len(pages), chunks=len(chunks))


@router.delete("/{name}", response_model=DeleteResponse)
def delete_document(name: str, request: Request) -> DeleteResponse:
    deleted = request.app.state.retriever.delete_document(safe_filename(name))
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DeleteResponse(document=name, deleted_chunks=deleted)
