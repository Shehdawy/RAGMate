"""Prompt construction shared by the API and the notebook (language-aware)."""
from app.services.language import is_arabic, refusal_for


def system_prompt(question: str) -> str:
    if is_arabic(question):
        language_rule = (
            "The question is in Arabic: write the whole answer in Arabic, keeping file names, "
            "numbers, technical terms and the [n] markers unchanged."
        )
    else:
        language_rule = "Answer in the same language as the question."
    return (
        "You are a document assistant. Answer the user's question using ONLY the numbered "
        "context passages provided. The passages may be written in a different language than the "
        "question: use their facts and never invent any. "
        f"{language_rule} Cite the passages you use with their numbers in square brackets, "
        "like [1] or [2][3]. If the context does not contain the answer, reply "
        f'exactly: "{refusal_for(question)}" Never use outside knowledge. '
        "Keep the answer concise (at most 5 sentences)."
    )


def user_prompt(question: str, context: str) -> str:
    tail = "Answer in Arabic, with [n] citations:" if is_arabic(question) else "Answer (with [n] citations):"
    return f"Context:\n{context}\n\nQuestion: {question}\n\n{tail}"


def build_messages(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt(question)},
        {"role": "user", "content": user_prompt(question, context)},
    ]
