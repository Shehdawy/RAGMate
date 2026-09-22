"""API-key authentication and rate limiting, applied as FastAPI dependencies."""
import hmac

from fastapi import Header, HTTPException, Request

from app.core.config import get_settings


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Require the X-API-Key header when API_KEY is configured (constant-time comparison)."""
    expected = get_settings().api_key
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def rate_limit(request: Request) -> None:
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    key = request.client.host if request.client else "unknown"
    allowed, retry_after = limiter.allow(key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a moment and try again.",
            headers={"Retry-After": str(retry_after)},
        )
