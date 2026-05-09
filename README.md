# MindCare AI — Intelligent Clinical Assistant by Mindbowser

> **AI Engineer Hackathon Assignment** | Mindbowser, Pune, India
> RAG-powered healthcare assistant that answers questions strictly from your uploaded documents.

---

## Demo Video

> **[▶ Watch Demo — MindCare AI by Mindbowser](https://github.com/Tanmayjagnade/Mindbowser_Assignment/releases/download/v1.0.0/mindbowser_assignment.mp4)**
>
> *MindCare AI Healthcare RAG Assistant — Full demo including document upload, RAG Q&A, appointment booking, and guardrails.*

---

## Overview

**MindCare AI** is a production-ready healthcare AI assistant built for Mindbowser clients. It ingests healthcare policy documents, stores them in a vector database, and answers questions using a local LLM — with source citations, confidence scoring, and guardrails against hallucination.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| Document Upload | Drag-and-drop .txt/.pdf/.md — shows chunk preview instantly |
| RAG Q&A | Answers only from your documents, never from outside knowledge |
| Source Citations | Every answer links back to the source document and chunk |
| Agentic Routing | 4-intent classifier: greeting / guardrail / appointment / knowledge |
| Guardrails | Blocks out-of-scope questions, refuses to hallucinate |
| Local LLM | Runs fully on-premise — no PHI leaves your network |
| React UI | ChatGPT-inspired interface with suggestion cards |

---

## Architecture

```
User Question
      |
POST /ask  (FastAPI)
      |
HealthcareAgent Router
      |
   +--+------------------+------------------+
   |                     |                  |
[Greeting]         [Appointment]       [Knowledge]
   |                     |                  |
Welcome msg   check_available_slots()   RAG Pipeline
                   mock tool                 |
                                    embed_query() [BGE-small]
                                             |
                                    ChromaDB cosine search
                                             |
                                    build_context()
                                             |
                                    LLM (local, on-premise)
                                    context-only prompt
                                             |
                               answer + sources + confidence
```

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| **API** | FastAPI | Fast, type-safe, auto Swagger docs |
| **Embedding** | `BAAI/bge-small-en-v1.5` | Retrieval-optimised, +3-5pt MTEB over MiniLM, asymmetric query prefix |
| **Vector DB** | ChromaDB | Embedded, cosine similarity, persistent, no extra server |
| **LLM** | Ollama (local) | On-premise, no API cost, HIPAA-aligned — no data leaves machine |
| **Frontend** | React (CDN) | Single HTML, no build step, ChatGPT-style UI |
| **Container** | Docker + Compose | Multi-stage build, Ollama sidecar, non-root user |

---

## Project Structure

```
healthcare-ai-assistant/
  app/
    main.py          # FastAPI app — all endpoints
    rag.py           # RAGService — ingest, retrieve, chunk, embed
    embeddings.py    # BGE embedding service with query prefix
    llm.py           # Multi-provider LLM (Ollama / OpenAI / Anthropic)
    agent.py         # Intent router + guardrails + appointment tool
    config.py        # Pydantic settings from .env
  data/              # 9 synthetic healthcare documents (no real PHI)
  static/
    index.html       # React UI (single file, CDN)
  tests/
    test_api.py      # Integration tests
  vector_store/      # ChromaDB — AUTO-CREATED at runtime, NOT in repo
  requirements.txt
  Dockerfile
  docker-compose.yml
  .env.example       # Template — copy to .env and fill in values
  TECHNICAL_REPORT.md
```

> **What is NOT committed to this repository (intentionally excluded):**
>
> | Excluded | Reason |
> |----------|--------|
> | `.env` | Contains secrets / API keys — never commit this |
> | `vector_store/` | Auto-generated at runtime by ChromaDB — re-created on `/ingest` |
> | `__pycache__/` | Python bytecode — auto-generated |
> | `.cache/` | HuggingFace model weights — downloaded automatically on first run |
>
> After cloning, run `cp .env.example .env`, configure it, then call `POST /ingest` to rebuild the vector store locally.

---

## Setup & Run

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed

### Step 1 — Install Ollama and pull a model

```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2:1b
ollama serve
```

### Step 2 — Clone and install

```bash
git clone https://github.com/Tanmayjagnade/Mindbowser_Assignment.git
cd Mindbowser_Assignment
pip install -r requirements.txt
```

### Step 3 — Configure

```bash
cp .env.example .env
# Edit .env if needed (default uses Ollama + llama3.2:1b)
```

### Step 4 — Run the server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 5 — Open in browser

| URL | Purpose |
|-----|---------|
| http://localhost:8000 | MindCare AI chat UI |
| http://localhost:8000/docs | Swagger API documentation |
| http://localhost:8000/health | Health check |

### Step 6 — Ingest documents

In the UI, click **"Load from /data folder"** or **drag and drop** a file.
Then start asking questions!

---

### Run with Docker

```bash
docker-compose up --build
# Pull model inside Ollama container (first time only):
docker exec ollama ollama pull llama3.2:1b
```

---

## API Reference

### POST `/ingest`
Bulk-ingest all documents from the `/data` folder.
```bash
curl -X POST http://localhost:8000/ingest \
     -H "Content-Type: application/json" -d '{}'
```
Response:
```json
{ "status": "success", "documents_processed": 9, "chunks_created": 101 }
```

### POST `/upload`
Upload a single file and see chunk preview.
```bash
curl -X POST http://localhost:8000/upload \
     -F "file=@my_document.pdf"
```
Response:
```json
{
  "status": "success",
  "filename": "my_document.pdf",
  "chunks_created": 12,
  "sample_chunks": [{ "index": 0, "text": "..." }]
}
```

### POST `/ask`
Ask a healthcare question.
```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Can a patient request a medication refill through telehealth?"}'
```
Response:
```json
{
  "answer": "Yes, medication refill reviews can be conducted during telehealth visits for non-controlled prescriptions...",
  "sources": [{ "document": "telehealth_guidelines.txt", "chunk": "Medication refill requests may be reviewed..." }],
  "confidence": "high",
  "intent": "knowledge_query",
  "tool_used": "rag_retrieval"
}
```

### POST `/ask` — Appointment booking
```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Book a cardiology appointment for Monday"}'
```
Response:
```json
{
  "answer": "Available slots for Cardiology on Monday, May 11 2026: 09:00 AM, 11:30 AM, 03:00 PM...",
  "intent": "appointment_booking",
  "tool_used": "check_available_slots",
  "tool_result": { "department": "Cardiology", "available_slots": ["09:00 AM", "11:30 AM"] }
}
```

### GET `/health`
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "chunks_indexed": 101
}
```

---

## Sample Questions to Try

| Question | Expected Behaviour |
|----------|-------------------|
| `What are my HIPAA rights?` | RAG — from hipaa_guidelines.txt |
| `How to request a medication refill?` | RAG — from medication_refill_policy.txt |
| `Can I use telehealth for a follow-up?` | RAG — from telehealth_guidelines.txt |
| `What diet after hospital discharge?` | RAG — from discharge_instructions.txt |
| `Book a cardiology slot for Monday` | Agent tool — mock appointment slots |
| `Who is the prime minister of India?` | Guardrail — blocked |
| `Hello!` | Greeting handler — welcome message |
| `What is the best cancer drug?` | Not found in documents |

---

## Technical Report

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for a detailed breakdown of:
- Embedding model choice (BGE vs MiniLM)
- Generation model (local LLM)
- Temperature and hyperparameters
- Guardrail implementation
- Intent classification approach
- Prompt injection mitigation

---

## Prompt Engineering

```
Read the healthcare document excerpts below and answer the question.
Use ONLY information from the excerpts. Do not add outside knowledge.
If the answer is not in the excerpts, say:
  "I could not find this information in the provided documents."
Never give a medical diagnosis or prescribe medication.

--- EXCERPTS ---
{retrieved_context}
--- END EXCERPTS ---

Question: {user_question}
Answer:
```

---

## Dataset

All 9 documents are **100% synthetic** — no real patient data or PHI:

| Document | Topic |
|----------|-------|
| discharge_instructions.txt | Post-hospital care, wound care, diet, activity |
| appointment_policy.txt | Scheduling, cancellation, wait times |
| insurance_faq.txt | Copay, deductible, eligibility, financial aid |
| hipaa_guidelines.txt | Patient rights, PHI, disclosures, privacy |
| medication_refill_policy.txt | Refill process, controlled substances |
| telehealth_guidelines.txt | Eligibility, tech requirements, medication via telehealth |
| privacy_guidelines.txt | Data handling, staff access, HIPAA-aligned |

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Future Improvements

1. **Streaming responses** — token-by-token output like ChatGPT (SSE)
2. **Conversation memory** — multi-turn context retention
3. **Voice interface** — speech-to-text + TTS for mobile
4. **FHIR R4 integration** — connect to EHR systems
5. **Drug interaction checker** — FDA openFDA API
6. **HIPAA audit trail** — immutable query log
7. **Role-based access** — doctor / nurse / patient views
8. **RLHF feedback loop** — thumbs up/down for fine-tuning
9. **Multi-language** — Hindi, Marathi, Tamil support
10. **Re-ranking** — cross-encoder for better chunk selection
11. **Hybrid search** — BM25 + semantic for medical keyword queries
12. **Named Entity Recognition** — highlight drug names, ICD-10 codes

---

## Security & Compliance

- No real PHI used anywhere
- Local LLM — data stays on-premise
- LLM refuses to diagnose or prescribe
- Non-root Docker container
- Secrets via `.env` (never hardcoded)
- HIPAA-aligned architecture (full compliance needs BAA + TLS + audit trail)

---

*Built by Tanmay Ambekar for the Mindbowser AI Engineer Hackathon Assignment*
