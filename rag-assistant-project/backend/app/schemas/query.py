from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        examples=["What is the F1 score?"],
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
