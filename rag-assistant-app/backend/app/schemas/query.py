from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    ref: int = Field(..., description="Passage number used as [n] in the answer")
    source: str
    page: int
    score: float = Field(..., description="Cosine similarity between question and passage")
    snippet: str


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        examples=["What is the F1 score?"],
    )
    top_k: int | None = Field(
        default=None, ge=1, le=10, description="Number of passages to retrieve (server default if omitted)"
    )

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("question must contain at least 3 non-space characters")
        return value


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    citations: list[Citation] = Field(default_factory=list)
    latency_ms: int | None = None
    model: str | None = None
