"""Application settings, loaded from environment variables / .env."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- application
    app_name: str = "RAG Document Assistant"
    environment: str = "development"
    enable_docs: bool = True            # set false in production to hide /docs and /openapi.json
    log_level: str = "INFO"
    log_format: str = "text"            # "text" or "json" (structured logs for production)
    cors_origins: str = "http://localhost:8501"  # comma-separated list

    # --- security
    api_key: str | None = None          # when set, every endpoint except /health, /ready, /metrics needs X-API-Key
    rate_limit_per_minute: int = 0      # per client, 0 disables rate limiting
    max_upload_mb: int = 20

    # --- language model: local Ollama or any OpenAI-compatible API (OpenAI, Groq, OpenRouter, vLLM, ...)
    llm_provider: str = "ollama"        # "ollama" or "openai"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str | None = None
    llm_model: str | None = None        # model name for the "openai" provider (falls back to ollama_model)
    llm_timeout_seconds: int = 120

    # --- retrieval
    top_k: int = 4
    min_similarity: float = 0.25
    vector_store_dir: Path = BACKEND_DIR / "data" / "vector_store"

    # --- indexing (used by `python -m app.cli ingest` and AUTO_INGEST)
    auto_ingest: bool = False           # build the vector store from raw documents when none exists
    raw_data_dir: Path | None = None    # defaults to backend/data/raw or <repo>/data/raw
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 800
    chunk_overlap: int = 100

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def resolved_llm_model(self) -> str:
        if self.llm_provider == "openai":
            return self.llm_model or self.ollama_model
        return self.ollama_model

    @property
    def resolved_raw_dir(self) -> Path:
        if self.raw_data_dir:
            return Path(self.raw_data_dir)
        for candidate in (BACKEND_DIR / "data" / "raw", BACKEND_DIR.parent / "data" / "raw"):
            if candidate.exists():
                return candidate
        return BACKEND_DIR / "data" / "raw"


@lru_cache
def get_settings() -> Settings:
    return Settings()
