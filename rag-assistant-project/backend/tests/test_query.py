import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.generation import (
    REFUSAL_MESSAGE,
    LLMUnavailableError,
    extract_citations,
    pick_sources,
)
from app.services.retrieval import RetrievedChunk

CHUNK = RetrievedChunk(
    text="The F1 score is the harmonic mean of precision and recall.",
    source="03_model_evaluation.pdf",
    page=1,
    score=0.8,
)


class FakeRetriever:
    chunk_count = 1

    def retrieve(self, question):
        return [CHUNK]


class FakeGenerator:
    def answer(self, question, chunks):
        return "F1 is the harmonic mean of precision and recall [1].", [chunks[0].label]


class DownGenerator:
    def answer(self, question, chunks):
        raise LLMUnavailableError("Ollama is down")


@pytest.fixture
def client():
    # TestClient used without `with` does not run the lifespan (no model loading).
    app.state.retriever = FakeRetriever()
    app.state.generator = FakeGenerator()
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_happy_path(client):
    response = client.post("/query", json={"question": "What is the F1 score?"})
    assert response.status_code == 200
    body = response.json()
    assert "harmonic mean" in body["answer"]
    assert body["sources"] == ["03_model_evaluation.pdf (page 1)"]


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "   "}, {}])
def test_query_invalid_input_returns_422(client, payload):
    response = client.post("/query", json=payload)
    assert response.status_code == 422


def test_query_llm_down_returns_503(client):
    app.state.generator = DownGenerator()
    response = client.post("/query", json={"question": "What is the F1 score?"})
    assert response.status_code == 503


def test_citation_parsing_and_sources():
    assert extract_citations("A [1] and B [2][3], also [1, 4] and [9].", 4) == [1, 2, 3, 4]
    assert pick_sources(REFUSAL_MESSAGE, [CHUNK]) == []
    assert pick_sources("No citation here.", [CHUNK]) == [CHUNK.label]
