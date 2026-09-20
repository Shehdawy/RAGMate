"""Application settings, loaded from environment variables / .env."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "RAG Document Assistant"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    vector_store_dir: Path = BACKEND_DIR / "data" / "vector_store"
    top_k: int = 4
    min_similarity: float = 0.25
    cors_origins: str = "http://localhost:8501"  # comma-separated list
    log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
