"""FastAPI application — Healthcare AI Assistant."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import HealthcareAgent
from app.config import get_settings, setup_logging
from app.rag import RAGService

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

cfg = get_settings()
setup_logging(cfg.log_level)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title=cfg.app_name,
    version=cfg.app_version,
    description=(
        "RAG-based healthcare AI assistant. Answers questions from healthcare "
        "policy documents using ChromaDB + sentence-transformers + Ollama/OpenAI/Anthropic."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    data_dir: Optional[str] = Field(
        default=None,
        description="Path to documents folder. Defaults to ./data",
    )


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_created: int
    collection: str


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Healthcare question to answer")


class SourceCitation(BaseModel):
    document: str
    chunk: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    confidence: str
    intent: Optional[str] = None
    tool_used: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_provider: str
    llm_model: str
    embedding_model: str
    vector_store: str
    chunks_indexed: int


# ---------------------------------------------------------------------------
# Lazy singleton agent
# ---------------------------------------------------------------------------

_agent: Optional[HealthcareAgent] = None


def _get_agent() -> HealthcareAgent:
    global _agent
    if _agent is None:
        logger.info("Initialising HealthcareAgent...")
        _agent = HealthcareAgent()
    return _agent


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest healthcare documents into the vector store",
    tags=["Ingestion"],
)
def ingest(request: IngestRequest = IngestRequest()) -> IngestResponse:
    """Read all .txt/.md/.pdf files from the data directory, chunk, embed, and store in ChromaDB."""
    logger.info("POST /ingest — data_dir=%s", request.data_dir)
    try:
        result = RAGService().ingest_documents(request.data_dir)
        global _agent
        _agent = None  # reset so agent picks up the fresh collection
        return IngestResponse(
            status="success",
            documents_processed=result["files_ingested"],
            chunks_created=result["chunks_ingested"],
            collection=cfg.chroma_collection_name,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected ingest error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a healthcare question (RAG + agentic routing)",
    tags=["Q&A"],
)
def ask(request: QuestionRequest) -> AskResponse:
    """
    Route through the agent:
    - Appointment questions -> check_available_slots mock tool
    - Knowledge questions   -> RAG pipeline (answer only from documents)
    """
    logger.info("POST /ask — question: %.80s", request.question)
    try:
        agent = _get_agent()
        result = agent.handle(request.question)
        return AskResponse(
            answer=result["answer"],
            sources=[SourceCitation(**s) for s in result.get("sources", [])],
            confidence=result.get("confidence", "none"),
            intent=result.get("intent"),
            tool_used=result.get("tool_used"),
            tool_result=result.get("tool_result"),
        )
    except Exception as exc:
        logger.exception("Error handling question")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["Health"],
)
def health() -> HealthResponse:
    """Returns service status and vector store statistics."""
    try:
        rag = RAGService()
        chunks = rag.collection.count()
        vector_store_status = "ok"
    except Exception:
        chunks = -1
        vector_store_status = "error"

    llm_model = (
        cfg.ollama_model if cfg.llm_provider == "ollama"
        else cfg.openai_model if cfg.llm_provider == "openai"
        else cfg.anthropic_model
    )

    return HealthResponse(
        status="healthy",
        version=cfg.app_version,
        llm_provider=cfg.llm_provider,
        llm_model=llm_model,
        embedding_model=cfg.embedding_model,
        vector_store=vector_store_status,
        chunks_indexed=chunks,
    )


# ---------------------------------------------------------------------------
# File upload endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/upload",
    summary="Upload a document, chunk it, embed it, and add to vector store",
    tags=["Ingestion"],
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a .txt / .pdf / .md file. The backend will:
    1. Save it to the data directory
    2. Split it into chunks
    3. Embed and store chunks in ChromaDB
    4. Return chunk count + first 3 sample chunks for preview
    """
    allowed = {".txt", ".pdf", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Allowed: .txt, .pdf, .md",
        )

    # Save file to data directory
    data_path = Path(cfg.data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    dest = data_path / file.filename

    content = await file.read()
    dest.write_bytes(content)
    logger.info("Saved uploaded file: %s (%d bytes)", dest, len(content))

    # Process through RAGService
    try:
        rag = RAGService()
        text = rag._read_file(dest)
        chunks = rag._split_text(text)

        if not chunks:
            raise ValueError("No text could be extracted from the file.")

        ids, docs, metas = [], [], []
        for idx, chunk in enumerate(chunks):
            ids.append(rag._chunk_id(file.filename, idx, chunk))
            docs.append(chunk)
            metas.append({"document": file.filename, "chunk_index": idx})

        vectors = rag.embedding_service.embed_documents(docs)
        rag.collection.upsert(ids=ids, embeddings=vectors, documents=docs, metadatas=metas)
        logger.info("Uploaded & indexed %s — %d chunks", file.filename, len(chunks))

        global _agent
        _agent = None  # reset so next /ask picks up the new document

        return {
            "status": "success",
            "filename": file.filename,
            "file_size_kb": round(len(content) / 1024, 1),
            "chunks_created": len(chunks),
            "sample_chunks": [
                {"index": i, "text": c[:300]} for i, c in enumerate(chunks[:4])
            ],
        }
    except Exception as exc:
        # Clean up saved file on failure
        if dest.exists():
            dest.unlink()
        logger.exception("Upload processing failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/documents",
    summary="List all indexed documents with chunk counts",
    tags=["Ingestion"],
)
def list_documents():
    """Returns all unique document names in the vector store with their chunk counts."""
    try:
        rag = RAGService()
        result = rag.collection.get(include=["metadatas"])
        counts: Dict[str, int] = {}
        for meta in result.get("metadatas", []):
            name = meta.get("document", "unknown")
            counts[name] = counts.get(name, 0) + 1
        docs = [{"filename": k, "chunks": v} for k, v in sorted(counts.items())]
        return {"documents": docs, "total_chunks": sum(counts.values())}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Delete document endpoint
# ---------------------------------------------------------------------------

@app.delete(
    "/documents/{filename}",
    summary="Delete a document and all its chunks from the vector store",
    tags=["Ingestion"],
)
def delete_document(filename: str):
    """
    Removes all chunks for the given document from ChromaDB
    and deletes the source file from the data directory.
    """
    try:
        rag = RAGService()
        result = rag.collection.get(where={"document": filename}, include=["metadatas"])
        ids = result.get("ids", [])
        if ids:
            rag.collection.delete(ids=ids)
            logger.info("Deleted %d chunks for document: %s", len(ids), filename)

        # Also remove the file from data directory
        file_path = Path(cfg.data_dir) / filename
        file_deleted = False
        if file_path.exists():
            file_path.unlink()
            file_deleted = True
            logger.info("Deleted file: %s", file_path)

        global _agent
        _agent = None  # reset agent after deletion

        return {
            "status": "success",
            "filename": filename,
            "chunks_deleted": len(ids),
            "file_deleted": file_deleted,
        }
    except Exception as exc:
        logger.exception("Delete failed for %s", filename)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Serve the React UI
# ---------------------------------------------------------------------------

import os as _os
_static_dir = _os.path.join(_os.path.dirname(__file__), "..", "static")
if _os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", include_in_schema=False)
def root():
    ui_path = _os.path.join(_os.path.dirname(__file__), "..", "static", "index.html")
    if _os.path.isfile(ui_path):
        return FileResponse(ui_path)
    return {"message": f"{cfg.app_name} is running", "docs": "/docs"}
