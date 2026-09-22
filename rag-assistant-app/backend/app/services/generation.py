"""Builds the grounded prompt and calls the configured LLM (blocking or streaming)."""
import logging
import re
from collections.abc import Iterator

from app.core.config import Settings
from app.services.llm import LLMClient, make_client
from app.services.language import REFUSAL_EN, is_refusal, refusal_for
from app.services.prompts import build_messages as build_prompt
from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = REFUSAL_EN  # kept for backwards compatibility; Arabic questions get REFUSAL_AR


class LLMUnavailableError(RuntimeError):
    """Raised when the Ollama server cannot produce an answer."""


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c.source}, page {c.page})\n{c.text}" for i, c in enumerate(chunks, 1)
    )


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    return build_prompt(question, format_context(chunks))


def extract_citations(answer: str, n_chunks: int) -> list[int]:
    """Return the sorted passage numbers cited in the answer, e.g. [1], [2][3], [1, 2]."""
    numbers: set[int] = set()
    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        for part in group.split(","):
            part = part.strip()
            if part.isdigit() and 1 <= int(part) <= n_chunks:
                numbers.add(int(part))
    return sorted(numbers)


def select_cited(answer: str, chunks: list[RetrievedChunk]) -> list[tuple[int, RetrievedChunk]]:
    """(passage number, chunk) pairs the answer relies on; all passages if it cites none."""
    if is_refusal(answer):
        return []
    cited = extract_citations(answer, len(chunks))
    if cited:
        return [(i, chunks[i - 1]) for i in cited]
    return list(enumerate(chunks, start=1))


def pick_sources(answer: str, chunks: list[RetrievedChunk]) -> list[str]:
    return list(dict.fromkeys(c.label for _, c in select_cited(answer, chunks)))


class Generator:
    """Grounded answer generation on top of any LLMClient (Ollama or an OpenAI-compatible API)."""

    def __init__(self, settings: Settings, client: LLMClient | None = None):
        self.client = client or make_client(settings)
        self.model = self.client.model

    def check_connection(self) -> bool:
        ok = self.client.ping()
        if ok:
            logger.info("Connected to the language model provider (model=%s)", self.model)
        return ok

    def _unavailable(self, exc: Exception) -> LLMUnavailableError:
        # Technical details go to the server log only; the user gets a plain message.
        logger.exception("LLM call failed (model=%s). Check the provider settings and that the model is available.", self.model)
        return LLMUnavailableError("The assistant is temporarily unavailable. Please try again in a moment.")

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
        if not chunks:
            return refusal_for(question), []
        try:
            answer = self.client.chat(build_messages(question, chunks)).strip()
        except Exception as exc:
            raise self._unavailable(exc) from exc
        return answer, pick_sources(answer, chunks)

    def stream(self, question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        """Yield the answer token by token."""
        if not chunks:
            yield refusal_for(question)
            return
        try:
            yield from self.client.stream(build_messages(question, chunks))
        except Exception as exc:
            raise self._unavailable(exc) from exc
