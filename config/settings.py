from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # SEC EDGAR
    sec_user_agent: str = "FinSight dev dev@example.com"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Scoring model
    base_model_id: str = "meta-llama/Llama-3.2-1B"
    lora_adapter_id: str = "likilogan/finsight-llama-lora-v1"
    hf_token: str | None = None
    model_device: str = "cpu"

    # Eval
    openai_api_key: str | None = None

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
