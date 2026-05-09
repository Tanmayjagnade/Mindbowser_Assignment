"""
Integration tests for the Healthcare AI Assistant API.
Run with:  pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "chunks_indexed" in data


# ---------------------------------------------------------------------------
# Ingest endpoint
# ---------------------------------------------------------------------------

def test_ingest_returns_success():
    resp = client.post("/ingest", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["documents_processed"] > 0
    assert data["chunks_created"] > 0


def test_ingest_idempotent():
    """Calling /ingest twice should not raise errors and produce same result."""
    resp1 = client.post("/ingest", json={})
    resp2 = client.post("/ingest", json={})
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json()["documents_processed"] == resp2.json()["documents_processed"]


def test_ingest_bad_dir_returns_404():
    resp = client.post("/ingest", json={"data_dir": "/nonexistent/path"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Ask endpoint — requires /ingest to have been called first
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def ingest_documents():
    """Ingest documents once before running Q&A tests."""
    resp = client.post("/ingest", json={})
    assert resp.status_code == 200


def test_ask_returns_expected_fields():
    resp = client.post("/ask", json={"question": "What is HIPAA?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data
    assert "confidence" in data
    assert "tool_used" in data


def test_ask_knowledge_question_uses_rag():
    resp = client.post("/ask", json={"question": "What are the rules for medication refills?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["tool_used"] == "rag_retrieval"
    assert data["confidence"] in ("high", "medium", "low", "none")


def test_ask_appointment_question_uses_tool():
    resp = client.post("/ask", json={"question": "Can I book a cardiology appointment for Monday?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "appointment_booking"
    assert data["tool_used"] == "check_available_slots"
    assert "available_slots" in data.get("tool_result", {})


def test_ask_out_of_scope_returns_not_found():
    resp = client.post("/ask", json={"question": "What is the capital of France?"})
    assert resp.status_code == 200
    answer = resp.json()["answer"].lower()
    assert "could not find" in answer


def test_ask_empty_question_returns_422():
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ask_sources_have_correct_structure():
    resp = client.post("/ask", json={"question": "What are HIPAA patient rights?"})
    assert resp.status_code == 200
    sources = resp.json().get("sources", [])
    for src in sources:
        assert "document" in src
        assert "chunk" in src
