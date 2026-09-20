"""Builds the grounded prompt and calls the local Ollama LLM."""
import logging
import re

import ollama

from app.core.config import Settings
from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = "I don't have enough information in the provided documents to answer that."

SYSTEM_PROMPT = (
    "You are a document assistant. Answer the user's question using ONLY the numbered "
    "context passages provided. Cite the passages you use with their numbers in square "
    "brackets, like [1] or [2][3]. If the context does not contain the answer, reply "
    f'exactly: "{REFUSAL_MESSAGE}" Never use outside knowledge. '
    "Keep the answer concise (at most 5 sentences)."
)


class LLMUnavailableError(RuntimeError):
    """Raised when the Ollama server cannot produce an answer."""


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c.source}, page {c.page})\n{c.text}" for i, c in enumerate(chunks, 1)
    )


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    user = (
        f"Context:\n{format_context(chunks)}\n\n"
        f"Question: {question}\n\nAnswer (with [n] citations):"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def extract_citations(answer: str, n_chunks: int) -> list[int]:
    """Return the sorted chunk numbers cited in the answer, e.g. [1], [2][3], [1, 2]."""
    numbers: set[int] = set()
    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        for part in group.split(","):
            part = part.strip()
            if part.isdigit() and 1 <= int(part) <= n_chunks:
                numbers.add(int(part))
    return sorted(numbers)


def pick_sources(answer: str, chunks: list[RetrievedChunk]) -> list[str]:
    if REFUSAL_MESSAGE in answer:
        return []
    cited = extract_citations(answer, len(chunks))
    used = [chunks[i - 1] for i in cited] or chunks
    return list(dict.fromkeys(c.label for c in used))


class Generator:
    def __init__(self, settings: Settings):
        self.client = ollama.Client(host=settings.ollama_host)
        self.model = settings.ollama_model

    def check_connection(self) -> bool:
        try:
            self.client.list()
            logger.info("Connected to Ollama, using model %s", self.model)
            return True
        except Exception as exc:  # startup must not crash if Ollama is still starting
            logger.warning("Ollama is not reachable yet: %s", exc)
            return False

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
        if not chunks:
            return REFUSAL_MESSAGE, []
        try:
            response = self.client.chat(
                model=self.model,
                messages=build_messages(question, chunks),
                options={"temperature": 0.1},
            )
        except Exception as exc:
            logger.exception("Ollama call failed")
            raise LLMUnavailableError(
                f"The language model is unavailable. Is Ollama running and is '{self.model}' pulled?"
            ) from exc
        answer = response["message"]["content"].strip()
        return answer, pick_sources(answer, chunks)
