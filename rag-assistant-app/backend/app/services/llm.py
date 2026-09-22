"""LLM clients behind one small interface: local Ollama or any OpenAI-compatible HTTP API."""
import json
import logging
from collections.abc import Iterator
from typing import Protocol

import requests

logger = logging.getLogger(__name__)
TEMPERATURE = 0.1


class LLMClient(Protocol):
    model: str

    def chat(self, messages: list[dict]) -> str: ...

    def stream(self, messages: list[dict]) -> Iterator[str]: ...

    def ping(self) -> bool: ...


class OllamaClient:
    provider = "ollama"

    def __init__(self, host: str, model: str):
        import ollama  # imported lazily so the "openai" provider does not need it

        self._client = ollama.Client(host=host)
        self.model = model

    def chat(self, messages: list[dict]) -> str:
        response = self._client.chat(model=self.model, messages=messages, options={"temperature": TEMPERATURE})
        return response["message"]["content"]

    def stream(self, messages: list[dict]) -> Iterator[str]:
        for part in self._client.chat(
            model=self.model, messages=messages, options={"temperature": TEMPERATURE}, stream=True
        ):
            token = part["message"]["content"]
            if token:
                yield token

    def ping(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception as exc:
            logger.warning("Ollama is not reachable: %s", exc)
            return False


class OpenAICompatClient:
    """Chat Completions API (POST {base_url}/chat/completions) with server-sent-event streaming."""

    provider = "openai"

    def __init__(self, base_url: str, api_key: str | None, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, messages: list[dict], stream: bool) -> dict:
        return {"model": self.model, "messages": messages, "temperature": TEMPERATURE, "stream": stream}

    def chat(self, messages: list[dict]) -> str:
        response = requests.post(
            f"{self.base_url}/chat/completions", json=self._payload(messages, False),
            headers=self._headers(), timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] or ""

    def stream(self, messages: list[dict]) -> Iterator[str]:
        with requests.post(
            f"{self.base_url}/chat/completions", json=self._payload(messages, True),
            headers=self._headers(), timeout=(10, self.timeout), stream=True,
        ) as response:
            response.raise_for_status()
            for raw in response.iter_lines(chunk_size=None):
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = (json.loads(data)["choices"][0].get("delta") or {}).get("content")
                except (ValueError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta

    def ping(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=5)
            return response.status_code < 400
        except requests.exceptions.RequestException as exc:
            logger.warning("LLM API is not reachable: %s", exc)
            return False


def make_client(settings) -> LLMClient:
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        return OllamaClient(settings.ollama_host, settings.resolved_llm_model)
    if provider == "openai":
        return OpenAICompatClient(
            settings.llm_base_url, settings.llm_api_key, settings.resolved_llm_model, settings.llm_timeout_seconds
        )
    raise ValueError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use 'ollama' or 'openai'.")
