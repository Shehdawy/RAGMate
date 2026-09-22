import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.routing import Match

from app.api.routes import documents, query
from app.core.config import get_settings
from app.core.metrics import metrics
from app.core.ratelimit import RateLimiter
from app.services.generation import Generator
from app.services.indexer import ensure_vector_store
from app.services.retrieval import Retriever
from app.utils.logging_config import setup_logging

settings = get_settings()
logger = logging.getLogger(__name__)
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare the vector store, then load the models and the LLM client once, at startup."""
    setup_logging(settings.log_level, settings.log_format)
    logger.info("Starting %s (environment=%s, llm_provider=%s)", settings.app_name, settings.environment, settings.llm_provider)
    ensure_vector_store(settings)
    app.state.retriever = Retriever(settings)
    app.state.generator = Generator(settings)
    if not app.state.generator.check_connection():
        logger.warning("The LLM provider is not reachable yet; /ready will report not ready until it is.")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    version="3.0.0",
    description=(
        "Retrieval-Augmented Generation API: hybrid retrieval (dense + BM25), grounded answers with "
        "citations in English and Arabic, token streaming and runtime document management. "
        "Works with a local Ollama model or any OpenAI-compatible API."
    ),
    openapi_tags=[
        {"name": "query", "description": "Ask questions and get grounded, cited answers"},
        {"name": "documents", "description": "Manage the knowledge base"},
        {"name": "system", "description": "Health, readiness and metrics"},
    ],
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.enable_docs else None,
    lifespan=lifespan,
)
app.state.rate_limiter = RateLimiter(settings.rate_limit_per_minute)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time-ms"],
)


def route_template(request: Request) -> str:
    """The matched route pattern (e.g. /documents/{name}); keeps metric labels bounded."""
    route = request.scope.get("route")
    if route is None:  # older Starlette versions do not expose it to middleware: match explicitly
        for candidate in request.app.routes:
            if candidate.matches(request.scope)[0] == Match.FULL:
                route = candidate
                break
    return getattr(route, "path", None) or "unmatched"


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    """Request id, timing, access log, metrics and basic security headers."""
    incoming = request.headers.get("X-Request-ID", "")
    request_id = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex[:12]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    metrics.observe_request(request.method, route_template(request), response.status_code, elapsed)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = f"{elapsed * 1000:.0f}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = response.headers.get("Cache-Control", "no-store")
    logger.info("%s %s -> %s in %.0f ms [%s]", request.method, request.url.path, response.status_code, elapsed * 1000, request_id)
    return response


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


app.include_router(query.router)
app.include_router(documents.router)
