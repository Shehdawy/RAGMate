"""Thin wrapper around the backend API."""
import requests


class APIError(Exception):
    """A friendly, user-facing error."""


def ask_question(base_url: str, question: str, timeout: int = 180) -> dict:
    url = f"{base_url.rstrip('/')}/query"
    try:
        response = requests.post(url, json={"question": question}, timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise APIError("Cannot reach the backend. Make sure it is running and API_BASE_URL is correct.")
    except requests.exceptions.Timeout:
        raise APIError("The request timed out. The language model may still be loading, please try again.")
    except requests.exceptions.RequestException as exc:
        raise APIError(f"Unexpected network error: {exc}")

    if response.status_code == 422:
        raise APIError("Please enter a valid question (3 to 1000 characters).")
    if response.status_code == 503:
        detail = response.json().get("detail", "The language model is unavailable.")
        raise APIError(detail)
    if not response.ok:
        raise APIError(f"The backend returned an error (HTTP {response.status_code}).")
    return response.json()


def check_health(base_url: str, timeout: int = 5) -> dict:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        raise APIError("Backend is not reachable.")
