"""Thin wrapper around the backend API.

Every failure becomes an APIError with a plain-language message that is safe to show to end users.
The raw technical detail is kept in `APIError.technical` (shown only when DEBUG_UI is enabled).
"""
import json
import os
from collections.abc import Iterator

import requests

UNAVAILABLE_MESSAGE = "The assistant is temporarily unavailable. Please try again in a moment."
TIMEOUT_MESSAGE = "This is taking longer than expected. Please try again."
GENERIC_MESSAGE = "Something went wrong. Please try again."


class APIError(Exception):
    """A friendly, user-facing error with optional technical detail for developers."""

    def __init__(self, message: str, technical: str | None = None):
        super().__init__(message)
        self.technical = technical


def _auth_headers() -> dict[str, str]:
    """The backend API key stays on the server side: Streamlit calls the API, browsers never see the key."""
    key = os.getenv("API_KEY")
    return {"X-API-Key": key} if key else {}


def _base(url: str) -> str:
    return url.rstrip("/")


def _detail(response: requests.Response) -> str | None:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return None
    return detail if isinstance(detail, str) else None


def _check(response: requests.Response) -> None:
    if response.ok:
        return
    detail = _detail(response)
    technical = f"HTTP {response.status_code} from {response.url}: {detail or response.text[:200]}"
    if response.status_code >= 500 or response.status_code in (401, 403):
        raise APIError(UNAVAILABLE_MESSAGE, technical)          # never expose server-side or auth error text
    if response.status_code == 422 and detail is None:
        raise APIError("Please enter a question between 3 and 1000 characters.", technical)
    raise APIError(detail or GENERIC_MESSAGE, technical)        # 4xx messages are written for users


def _request(method: str, url: str, **kwargs) -> requests.Response:
    headers = {**_auth_headers(), **kwargs.pop("headers", {})}
    try:
        return requests.request(method, url, headers=headers, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise APIError(UNAVAILABLE_MESSAGE, f"Cannot connect to {url}: {exc}")
    except requests.exceptions.Timeout as exc:
        raise APIError(TIMEOUT_MESSAGE, f"Timeout calling {url}: {exc}")
    except requests.exceptions.RequestException as exc:
        raise APIError(GENERIC_MESSAGE, f"{url}: {exc}")


def get_health(base_url: str, timeout: int = 3) -> dict | None:
    """Backend status, or None if it is unreachable."""
    try:
        response = requests.get(f"{_base(base_url)}/health", headers=_auth_headers(), timeout=timeout)
        return response.json() if response.ok else None
    except (requests.exceptions.RequestException, ValueError):
        return None


def stream_query(base_url: str, question: str, top_k: int | None = None, state: dict | None = None) -> Iterator[str]:
    """Yield the answer token by token.

    When the stream ends, `state` is filled with citations, latency_ms and model.
    """
    state = state if state is not None else {}
    payload = {"question": question, **({"top_k": top_k} if top_k else {})}
    response = _request(
        "POST", f"{_base(base_url)}/query/stream", json=payload, stream=True, timeout=(5, 300)
    )
    _check(response)
    response.encoding = "utf-8"
    for raw in response.iter_lines(chunk_size=None):
        if not raw:
            continue
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind = event.get("type")
        if kind == "token":
            yield event.get("text", "")
        elif kind == "done":
            state.update(
                citations=event.get("citations", []),
                sources=event.get("sources", []),
                latency_ms=event.get("latency_ms"),
                model=event.get("model"),
            )
        elif kind == "error":
            raise APIError(UNAVAILABLE_MESSAGE, event.get("detail"))


def list_documents(base_url: str) -> list[dict]:
    response = _request("GET", f"{_base(base_url)}/documents", timeout=10)
    _check(response)
    return response.json()


def upload_document(base_url: str, filename: str, data: bytes) -> dict:
    response = _request(
        "POST", f"{_base(base_url)}/documents", files={"file": (filename, data)}, timeout=(5, 600)
    )
    _check(response)
    return response.json()


def delete_document(base_url: str, name: str) -> dict:
    response = _request("DELETE", f"{_base(base_url)}/documents/{requests.utils.quote(name, safe='')}", timeout=30)
    _check(response)
    return response.json()
