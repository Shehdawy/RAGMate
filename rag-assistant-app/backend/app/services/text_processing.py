"""Text cleaning and chunking (shared by document upload and the notebook logic)."""
import re

SENTENCE_ENDS = (". ", "? ", "! ")


def clean_text(text: str) -> str:
    """Normalize text extracted from a PDF page."""
    text = re.sub(r"-\n(\w)", r"\1", text)   # re-join words hyphenated across lines
    text = re.sub(r"\s*\n\s*", " ", text)     # line breaks -> spaces
    text = re.sub(r"\s{2,}", " ", text)       # collapse repeated whitespace
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Fixed-size character chunks with overlap that prefer to end at a sentence boundary."""
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window_start = start + int(chunk_size * 0.7)          # look only in the last 30 percent
            cut = max(text.rfind(t, window_start, end) for t in SENTENCE_ENDS)
            if cut != -1:
                end = cut + 1                                     # keep the punctuation mark
            else:
                space = text.rfind(" ", window_start, end)
                if space != -1:
                    end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        space = text.find(" ", next_start, end)                   # start the next chunk on a word boundary
        start = space + 1 if space != -1 else next_start
    return chunks
