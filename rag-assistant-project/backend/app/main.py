import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router
from app.core.config import get_settings
from app.services.generation import Generator
from app.services.retrieval import Retriever
from app.utils.logging_config import setup_logging

settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector store and the LLM client once, at startup."""
    setup_logging(settings.log_level)
    logger.info("Starting %s", settings.app_name)
    app.state.retriever = Retriever(settings)
    app.state.generator = Generator(settings)
    app.state.generator.check_connection()
    yield
    logger.info("Shutting down")


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
