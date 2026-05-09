import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import chromadb
from chromadb.api.models.Collection import Collection
from pypdf import PdfReader

from app.config import get_settings
from app.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    document: str
    chunk: str
    score: float


class RAGService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        self.client = chromadb.PersistentClient(path=self.settings.chroma_persist_dir)
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_documents(self, data_dir: str | None = None) -> dict:
        source_dir = Path(data_dir or self.settings.data_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {source_dir}")

        files = [p for p in source_dir.rglob("*") if p.suffix.lower() in {".txt", ".md", ".pdf"}]
        if not files:
            raise ValueError("No supported files found in data directory")

        total_chunks = 0
        upsert_ids: List[str] = []
        docs: List[str] = []
        metas: List[dict] = []

        for path in files:
            text = self._read_file(path)
            chunks = self._split_text(text)
            for idx, chunk in enumerate(chunks):
                chunk_id = self._chunk_id(path.name, idx, chunk)
                upsert_ids.append(chunk_id)
                docs.append(chunk)
                metas.append({"document": path.name, "chunk_index": idx})

            total_chunks += len(chunks)
            logger.info("Prepared %d chunks from %s", len(chunks), path.name)

        vectors = self.embedding_service.embed_documents(docs)
        self.collection.upsert(ids=upsert_ids, embeddings=vectors, documents=docs, metadatas=metas)
        logger.info("Ingestion complete. Files=%d Chunks=%d", len(files), total_chunks)
        return {"files_ingested": len(files), "chunks_ingested": total_chunks}

    def retrieve(self, question: str, top_k: int | None = None) -> List[RetrievedChunk]:
        k = top_k or self.settings.top_k_results
        query_embedding = self.embedding_service.embed_query(question)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: List[RetrievedChunk] = []
        for doc_text, meta, distance in zip(documents, metadatas, distances):
            score = max(0.0, 1.0 - float(distance))
            if score < self.settings.min_similarity_score:
                continue
            chunks.append(
                RetrievedChunk(
                    document=str(meta.get("document", "unknown")),
                    chunk=doc_text,
                    score=score,
                )
            )
        return chunks

    def build_context(self, retrieved: List[RetrievedChunk]) -> str:
        parts = []
        for idx, item in enumerate(retrieved, 1):
            parts.append(f"[Source {idx} | {item.document}]\n{item.chunk}")
        return "\n\n".join(parts)

    def estimate_confidence(self, retrieved: List[RetrievedChunk]) -> str:
        if not retrieved:
            return "low"
        avg = sum(r.score for r in retrieved) / len(retrieved)
        if avg >= 0.75:
            return "high"
        if avg >= 0.55:
            return "medium"
        return "low"

    def _chunk_id(self, filename: str, idx: int, text: str) -> str:
        signature = hashlib.sha256(f"{filename}:{idx}:{text}".encode("utf-8")).hexdigest()[:16]
        return f"{filename}-{idx}-{signature}"

    def _split_text(self, text: str) -> List[str]:
        text = " ".join(text.split())
        if not text:
            return []
        size = self.settings.chunk_size
        overlap = self.settings.chunk_overlap
        step = max(1, size - overlap)
        chunks = []
        for start in range(0, len(text), step):
            chunk = text[start : start + size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _read_file(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8")
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        raise ValueError(f"Unsupported file type: {path}")
