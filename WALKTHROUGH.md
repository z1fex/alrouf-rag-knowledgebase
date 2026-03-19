# Walkthrough Guide

Companion document for evaluators reviewing the Alrouf RAG Knowledge Base.

---

## Overview

This system is a Retrieval-Augmented Generation (RAG) Q&A application built for Alrouf Lighting Technology. It ingests 5 bilingual company documents, indexes them in a FAISS vector database, and answers user questions with source citations. The system supports both Arabic and English, detects out-of-scope questions, and runs fully offline without API keys in mock mode.

---

## Architecture — Data Flow

```
User types question
        │
        ▼
[1] Query hits POST /api/query (FastAPI)
        │
        ▼
[2] Query text is embedded using the same embedding service
    that was used to index the documents
    - Mock: TF-IDF character n-gram vectorizer + SVD reduction
    - Live: OpenAI text-embedding-3-small API call
        │
        ▼
[3] FAISS IndexFlatIP searches for the 3 most similar chunk
    vectors using cosine similarity (inner product on
    L2-normalized vectors)
        │
        ▼
[4] Scope checker evaluates whether the query is in-scope:
    - If best similarity < 0.3 → out of scope
    - If query contains domain keywords → in scope
    - Otherwise → out of scope (prevents false positives)
        │
        ├── Out of scope → return polite refusal (EN or AR)
        │
        ▼
[5] Generator produces the answer:
    - Mock: pattern-matches query keywords against 12
      pre-written responses in sample_responses.json
    - Live: builds a prompt with system instructions +
      retrieved chunks, sends to gpt-4o-mini
        │
        ▼
[6] Response returned with:
    - Answer text
    - Citations (document title, chunk index, score)
    - Scope status
    - Latency in milliseconds
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **No LangChain** | Built pipeline from scratch to demonstrate understanding of RAG internals. For 5 documents, LangChain adds unnecessary abstraction. |
| **FAISS over ChromaDB** | Simpler (no server), serializes to 2 files, exact search is fast enough for 31 chunks. |
| **TF-IDF mock embeddings** | Works offline, handles Arabic via character n-grams, no PyTorch dependency (~30MB vs ~2GB). |
| **FastAPI over Streamlit** | Proper REST API, Pydantic validation, static file serving, production-ready with uvicorn. |
| **Character n-grams (2-4)** | Handles Arabic morphology without a dedicated tokenizer library. Captures subword patterns across languages. |
| **Two-layer scope check** | Similarity alone fails with TF-IDF (common substrings like city names cause false positives). Adding keyword heuristic gives reliable filtering. |

---

## How to Test — Step by Step

### 1. Setup (2 minutes)

```bash
cd alrouf-rag-knowledgebase
python -m venv venv
source venv/bin/activate    # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
python -m scripts.build_index
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 2. Open the Web UI

Navigate to `http://127.0.0.1:8000` in a browser.

### 3. Test In-Scope English Query

Click the **"IP rating of ALR-SL-90W"** example button, or type:
> What is the IP rating of the ALR-SL-90W?

**Expected**: Answer mentions IP66, citations from Product Catalog, green "In Scope" badge.

### 4. Test Installation Query

Click **"Installation guide"** or type:
> How do I install a streetlight?

**Expected**: Step-by-step installation instructions, citations from Installation & Maintenance Guide.

### 5. Test Warranty Query

Type:
> What is the warranty period for LED components?

**Expected**: Answer mentions 5 years LED, 10 years structure, citations from Warranty Policy.

### 6. Test Arabic Query

Click **"شهادات الجودة"** or select "العربية" and type:
> ما هي شهادات الجودة لشركة الروف؟

**Expected**: Arabic response about ISO certifications and SASO, RTL text layout, citations from Company Profile.

### 7. Test Out-of-Scope Query

Click **"Out-of-scope test"** or type:
> What is the weather in Riyadh?

**Expected**: Polite refusal message, red "Out of Scope" badge, no citations.

### 8. Test API Directly

```bash
# Health check
curl http://localhost:8000/api/health

# English query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What projects has Alrouf completed?", "language": "en"}'
```

### 9. Run Tests

```bash
python -m pytest tests/ -v
# Expected: 43 passed
```

---

## What to Show in a Video Demo

1. **Start the server** — show the terminal output with "Index ready. Mode: MOCK"
2. **Open the web UI** — show the clean interface with Alrouf branding and "MOCK MODE" badge
3. **English query** — ask about ALR-SL-90W specs, show the answer with citations
4. **Arabic query** — switch to Arabic, ask about certifications, show RTL text layout
5. **Out-of-scope** — ask "What is the weather?", show the red "Out of Scope" badge and refusal
6. **API call** — show a curl command and JSON response in the terminal
7. **Run tests** — show `pytest -v` with all 43 tests passing
8. **Show code** — briefly scroll through `ingest.py`, `embeddings.py`, `generator.py` to highlight the pipeline
9. **Show project structure** — `ls` the project tree to show completeness
