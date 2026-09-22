from pydantic import BaseModel


class DocumentInfo(BaseModel):
    name: str
    chunks: int


class UploadResponse(BaseModel):
    document: str
    pages: int
    chunks: int


class DeleteResponse(BaseModel):
    document: str
    deleted_chunks: int
