import logging
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "Healthcare AI Assistant"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # LLM — set LLM_PROVIDER=openai|anthropic|ollama in .env
    llm_provider: str = "ollama"
    llm_temperature: float = 0.1   # low temp = factual, deterministic answers
    llm_max_tokens: int = 512     # sufficient for RAG answers; keeps latency low
    llm_num_ctx: int = 4096       # Ollama context window (fits prompt + chunks + answer)

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    # Anthropic
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # Ollama (local — no API key needed, bonus points)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # Embeddings — BGE model, runs locally, no API key required
    # BAAI/bge-small-en-v1.5: 384-dim, retrieval-optimised, outperforms MiniLM on MTEB
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # ChromaDB
    chroma_persist_dir: str = "./vector_store"
    chroma_collection_name: str = "healthcare_docs"

    # RAG
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k_results: int = 5
    min_similarity_score: float = 0.30

    # Data
    data_dir: str = "./data"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
