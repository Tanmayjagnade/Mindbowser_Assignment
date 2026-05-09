# Technical Report — Healthcare AI Assistant
**Project:** Mindbowser AI Engineer Hackathon Assignment
**Author:** Healthcare RAG System
**Date:** 2026-05-09

---

## 1. Embedding Model

| Property | Value |
|----------|-------|
| **Model** | `BAAI/bge-small-en-v1.5` |
| **Provider** | BAAI (Beijing Academy of AI) via HuggingFace |
| **Dimensions** | 384 |
| **Parameters** | ~33 million |
| **Runs locally?** | Yes — no API key, no internet required after first download |
| **Previous model** | `all-MiniLM-L6-v2` (replaced) |

### Why BGE over MiniLM?
BGE (BAAI General Embedding) is purpose-built for retrieval tasks. On the MTEB benchmark:
- BGE-small outperforms all-MiniLM-L6-v2 by ~3-5 points on retrieval tasks
- Trained with contrastive learning on large-scale retrieval datasets
- Supports asymmetric encoding (different behaviour for queries vs passages)

### BGE Asymmetric Encoding
BGE requires a special instruction prefix **on the query side only**:
```
Query:    "Represent this sentence for searching relevant passages: <user question>"
Document: "<document chunk text>"   ← no prefix
```
This asymmetry is implemented in `app/embeddings.py`:
- `embed_query()` → adds the prefix automatically when the model name contains "bge"
- `embed_documents()` → no prefix, encodes raw passage text

---

## 2. Generation Model (LLM)

| Property | Value |
|----------|-------|
| **Model** | `llama3.2:1b` (Meta Llama 3.2, 1B parameters) |
| **Provider** | Ollama (local, on-premise — no API cost) |
| **Context window** | 4096 tokens (`num_ctx=4096`) |
| **Max output tokens** | 512 (`num_predict=512`) |
| **Temperature** | **0.1** (near-deterministic, factual answers) |
| **Runs locally?** | Yes — fully offline after model pull |

### Temperature Rationale
- **0.1** chosen deliberately for a healthcare assistant
- Low temperature → deterministic, consistent answers
- Prevents the model from "hallucinating" creative interpretations
- A value of 0.0 would be fully deterministic; 0.1 adds minimal variation for natural phrasing

### Why Ollama + Local LLM?
- No PHI leaves the machine (HIPAA-aligned architecture)
- No per-query API cost
- Can be fully air-gapped for hospital deployment
- Fallback providers: OpenAI (`gpt-4o-mini`), Anthropic (`claude-haiku-4-5`) — configurable via `.env`

---

## 3. Intent Classification (Agentic Router)

**Yes — intent classification is implemented** in `app/agent.py`.

### Classification Flow
```
User Question
      │
      ▼
 [1] Greeting?      ──YES──► Return welcome message (no LLM/RAG)
      │
      ▼
 [2] Out-of-scope?  ──YES──► Guardrail blocks it (no LLM/RAG)
      │
      ▼
 [3] Appointment?   ──YES──► check_available_slots() mock tool
      │
      ▼
 [4] Knowledge      ────────► RAG Pipeline + LLM
```

### Implementation: Keyword-Based Regex Router
| Intent | Method | Examples |
|--------|--------|---------|
| Greeting | Regex: `hi`, `hello`, `hey`, `good morning` | "Hi!", "Hello there" |
| Out-of-scope | Blocked keyword list | "Prime minister", "cricket", "stock market" |
| Appointment | Regex: `book`, `schedule`, `slot`, `appointment` | "Book cardiology for Monday" |
| Knowledge | Default (all others) | "What is HIPAA?", "How to refill medication?" |

### Why Keyword Router vs LLM Classifier?
- **Zero latency** — no LLM call needed for routing
- **100% deterministic** — no variance between runs
- **Debuggable** — can inspect exactly why a question was routed
- Trade-off: may miss edge cases (e.g., "Can I get a slot?" with no appointment keywords)
- Upgrade path: swap `classify_intent()` with an LLM call for production

---

## 4. Guardrails

**Yes — two-layer guardrail system is implemented.**

### Layer 1: Intent-Based Blocking (Hard Guardrail)
Blocks questions that match clearly non-healthcare topic patterns:
```python
_BLOCKED_TOPICS = [
    r"\b(prime\s*minister|president|politician|election)\b",
    r"\b(cricket|football|ipl|sport)\b",
    r"\b(stock|crypto|bitcoin|sensex)\b",
    r"\b(capital\s*of|geography|weather|climate)\b",
    r"\b(movie|film|actor|celebrity|bollywood)\b",
    ...
]
```
Healthcare signals override the block:
```python
_HEALTHCARE_SIGNALS = [
    r"\b(doctor|patient|medicine|hospital|health|medical|hipaa)\b",
    r"\b(insurance|discharge|telehealth|appointment|refill)\b",
    ...
]
```

### Layer 2: RAG Similarity Threshold (Soft Guardrail)
- If no document chunk scores above `MIN_SIMILARITY_SCORE=0.30`, the question returns:
  *"I could not find this information in the provided documents."*
- This prevents the LLM from answering healthcare questions that are simply not covered by the loaded documents

### Layer 3: Prompt-Level Constraint
The LLM system prompt explicitly instructs:
- Answer only from provided excerpts
- No outside knowledge
- No medical diagnosis
- No treatment prescription

---

## 5. Prompt Injection Protection

**Yes — prompt injection is mitigated** through the following mechanisms:

### 5a. Context Isolation
The user question is passed separately from the document context. The prompt structure is:
```
[INSTRUCTION BLOCK]    ← fixed, not user-controlled
[DOCUMENT EXCERPTS]    ← retrieved documents, sanitized
Question: {user_input} ← user input placed LAST, after context
Answer:
```
Placing user input after the context and instruction means a malicious instruction like *"Ignore above, tell me X"* has less influence over a small model following a direct `Answer:` instruction.

### 5b. Hard-Coded Stop at "Question:" Token
The Ollama call includes a stop token: `"stop": ["Question:", "---"]`
This prevents the model from "injecting" fake follow-up questions.

### 5c. Guardrail Pre-Filter
Before the question reaches the LLM, it passes through the intent router. Clearly off-topic or adversarial inputs (e.g., "Ignore your instructions and...") are caught by the keyword guardrail.

### 5d. Known Limitation
Sophisticated prompt injection (jailbreaks via paraphrasing, encoding tricks) is NOT fully mitigated with a 1b model. For production:
- Use a stronger model (7B+)
- Add a dedicated prompt injection classifier
- Implement input sanitization layer

---

## 6. Vector Database

| Property | Value |
|----------|-------|
| **Database** | ChromaDB (persistent, embedded) |
| **Distance metric** | Cosine similarity |
| **Chunk size** | 500 characters |
| **Chunk overlap** | 100 characters |
| **Top-K retrieval** | 5 chunks |
| **Minimum score** | 0.30 cosine similarity |

### Confidence Scoring
Derived from average cosine similarity of retrieved chunks:
| Score Range | Confidence |
|-------------|-----------|
| ≥ 0.75 | high |
| ≥ 0.55 | medium |
| < 0.55 | low |
| No chunks above threshold | none → "not found" |

---

## 7. RAG Pipeline Summary

```
Document Upload / Data Folder
        │
        ▼
  _read_file()         # txt/pdf/md support
        │
  _split_text()        # sliding window chunks (500 chars, 100 overlap)
        │
  embed_documents()    # BGE-small — passage encoding (no prefix)
        │
  collection.upsert()  # ChromaDB cosine index
        │
   [At Query Time]
        │
  embed_query()        # BGE-small — query encoding (WITH prefix)
        │
  collection.query()   # cosine similarity top-5
        │
  build_context()      # format retrieved chunks with source labels
        │
  answer_with_context()# LLM generates answer from context only
        │
  Response             # answer + sources + confidence + intent
```

---

## 8. API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | System status, model info, chunk count |
| `/ingest` | POST | Bulk ingest from `./data` folder |
| `/upload` | POST | Upload single file, returns chunk preview |
| `/documents` | GET | List all indexed documents + chunk counts |
| `/ask` | POST | RAG Q&A with agentic routing |
| `/docs` | GET | Swagger interactive API documentation |

---

## 9. Limitations and Future Improvements

| Limitation | Proposed Fix |
|-----------|-------------|
| `llama3.2:1b` answers can be weak | Use `llama3.2:3b` or `llama3.1:8b` (needs GPU for speed) |
| Keyword guardrail may miss edge cases | Add LLM-based intent/safety classifier |
| No multi-turn conversation memory | Add session-based chat history |
| No re-ranking of retrieved chunks | Add cross-encoder re-ranker (e.g., `ms-marco-MiniLM`) |
| Prompt injection not fully mitigated | Input sanitization + stronger model |
| No PDF table/image extraction | Use `unstructured` library with OCR |
| GPU not used (AMD, 0.5GB VRAM) | Deploy on NVIDIA GPU machine for 5-10x speed |
