"""
Embedding service using BAAI/bge-small-en-v1.5 (BGE model).

BGE (BAAI General Embedding) is specifically trained for retrieval tasks and
outperforms all-MiniLM-L6-v2 on MTEB benchmarks, especially for RAG use cases.

Key BGE requirement: queries must be prefixed with an instruction string.
Documents are embedded as-is (no prefix). This asymmetric encoding improves
retrieval precision significantly.
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)

# BGE requires this prefix on queries (NOT on documents) for optimal retrieval
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model_instance: SentenceTransformer | None = None


def _is_bge(model_name: str) -> bool:
    return "bge" in model_name.lower()


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the embedding model as a singleton."""
    global _model_instance
    if _model_instance is None:
        model_name = get_settings().embedding_model
        logger.info("Loading embedding model: %s", model_name)
        _model_instance = SentenceTransformer(model_name)
        dim = _model_instance.get_sentence_embedding_dimension()
        logger.info("Embedding model ready — dim=%d  bge_prefix=%s", dim, _is_bge(model_name))
    return _model_instance


class EmbeddingService:
    """
    Embedding service backed by sentence-transformers.

    Model: BAAI/bge-small-en-v1.5 (default)
      - 384 dimensions
      - Trained for asymmetric retrieval (query vs. passage)
      - Runs fully locally, no API key required
      - ~33 MB model size, fast on CPU
    """

    def __init__(self) -> None:
        self.model = get_embedding_model()
        self.model_name = get_settings().embedding_model
        self._use_bge_prefix = _is_bge(self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed document chunks — no instruction prefix for BGE passage encoding."""
        logger.debug("Embedding %d chunks with %s", len(texts), self.model_name)
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a user query.
        BGE models require an instruction prefix on the query side only.
        """
        query = (_BGE_QUERY_PREFIX + text) if self._use_bge_prefix else text
        vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector[0].tolist()

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
