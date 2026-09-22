"""Turn an uploaded file into page texts and chunks ready for indexing."""
import io
from pathlib import Path

from app.services.text_processing import chunk_text, clean_text

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}
MIN_PAGE_CHARS = 50  # pages with less text are treated as scanned images


class IngestionError(ValueError):
    """The uploaded file cannot be indexed (message is safe to show to the user)."""


def safe_filename(name: str | None) -> str:
    return Path(name or "").name.strip()


def extract_pages(filename: str, data: bytes) -> list[dict]:
    suffix = Path(filename).suffix.lower()
    pages: list[dict] = []
    if suffix == ".pdf":
        from pypdf import PdfReader  # imported lazily

        try:
            reader = PdfReader(io.BytesIO(data))
            for number, page in enumerate(reader.pages, start=1):
                text = clean_text(page.extract_text() or "")
                if len(text) >= MIN_PAGE_CHARS:
                    pages.append({"page": number, "text": text})
        except Exception as exc:
            raise IngestionError("This PDF could not be read. It may be damaged or password-protected.") from exc
    elif suffix in {".txt", ".md"}:
        text = clean_text(data.decode("utf-8", errors="ignore"))
        if len(text) >= MIN_PAGE_CHARS:
            pages.append({"page": 1, "text": text})
    else:
        raise IngestionError("This file type isn't supported. Please upload a PDF, TXT or MD file.")
    if not pages:
        raise IngestionError("No readable text was found in this file. Scanned documents need to be converted to text first.")
    return pages


def build_chunks(filename: str, pages: list[dict], chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    chunks: list[dict] = []
    for page in pages:
        for i, piece in enumerate(chunk_text(page["text"], chunk_size, overlap)):
            chunks.append({
                "id": f"{filename}-p{page['page']}-c{i}",
                "source": filename,
                "page": page["page"],
                "text": piece,
            })
    return chunks
