# Performance & Cost Report

## Index Statistics

| Metric | Value |
|---|---|
| Documents indexed | 5 |
| Total chunks | 31 |
| Chunk size target | 800 tokens (with 100-token overlap) |
| Chunk distribution | 5-7 chunks per document |
| Index type | FAISS `IndexFlatIP` (exact brute-force search) |
| Mock embedding dimensions | 30 (TF-IDF + TruncatedSVD, fitted to corpus) |
| Live embedding dimensions | 1,536 (OpenAI `text-embedding-3-small`) |
| FAISS index file size (mock) | 3.7 KB |
| Metadata file size (mock) | 55 KB |
| TF-IDF model file size | 2.7 MB |
| Total disk footprint (mock) | ~2.8 MB |

### Chunk Distribution by Document

| Document | ID | Chunks |
|---|---|---|
| Alrouf Product Catalog 2024 | DOC-001 | 6 |
| Installation & Maintenance Guide | DOC-002 | 6 |
| Company Profile & Certifications | DOC-003 | 5 |
| Warranty & After-Sales Policy | DOC-004 | 7 |
| Technical Specifications — Street Lighting | DOC-005 | 7 |

---

## Query Latency

All measurements are wall-clock end-to-end for the `POST /api/query` endpoint, measured locally.

### Mock Mode (USE_MOCK_LLM=true)

| Stage | Average Latency |
|---|---|
| Query embedding (TF-IDF transform) | ~1-2 ms |
| FAISS search (top-3, 31 vectors) | <0.5 ms |
| Scope check (keyword matching) | <0.1 ms |
| Response generation (pattern match) | ~1 ms |
| **Total per query** | **~3-8 ms** |

### Live Mode (USE_MOCK_LLM=false, estimated)

| Stage | Average Latency |
|---|---|
| Query embedding (OpenAI API call) | ~200-400 ms |
| FAISS search (top-3) | <0.5 ms |
| Scope check | <0.1 ms |
| LLM generation (OpenAI API call) | ~800-2,000 ms |
| **Total per query** | **~1,000-2,500 ms** |

> **Note:** Live mode latency is dominated by network round-trips to the OpenAI API. Actual latency varies with API load, network conditions, and response length.

### Comparison

| Metric | Mock Mode | Live Mode |
|---|---|---|
| Average query latency | ~5 ms | ~1,500 ms |
| P99 query latency | ~15 ms | ~3,000 ms |
| Requires network | No | Yes |
| Requires API key | No | Yes |

---

## Cost Analysis (Live Mode)

Based on OpenAI pricing (as of early 2025):

### Embedding Cost

| Model | Price | Tokens per Query | Cost per Query |
|---|---|---|---|
| `text-embedding-3-small` | $0.020 / 1M tokens | ~20-50 tokens | ~$0.000001 |

### Generation Cost

| Model | Input Price | Output Price | Avg Input Tokens | Avg Output Tokens | Cost per Query |
|---|---|---|---|---|---|
| `gpt-4o-mini` | $0.15 / 1M tokens | $0.60 / 1M tokens | ~2,000 (system prompt + 3 chunks) | ~300 (answer) | ~$0.00048 |

### Total Cost Summary

| Scenario | Cost per Query | Cost per 1,000 Queries | Cost per 10,000 Queries |
|---|---|---|---|
| **Mock mode** | $0.00 | $0.00 | $0.00 |
| **Live mode** | ~$0.0005 | ~$0.50 | ~$5.00 |

The embedding cost is negligible (~0.2% of total). Generation dominates at ~99.8% of per-query cost.

---

## Index Build Performance

### Mock Mode

| Stage | Time |
|---|---|
| Document loading + chunking | ~1.2 s |
| TF-IDF vectorizer fitting | ~0.05 s |
| Embedding generation (transform) | ~0.07 s |
| FAISS index building + save | <0.01 s |
| **Total** | **~1.3 s** |

### Live Mode (estimated)

| Stage | Time |
|---|---|
| Document loading + chunking | ~1.2 s |
| Embedding generation (OpenAI API, 31 chunks batched) | ~3-5 s |
| FAISS index building + save | <0.01 s |
| **Total** | **~4-6 s** |

---

## Production Optimization Recommendations

1. **Index type upgrade** — For datasets exceeding 10,000 chunks, switch from `IndexFlatIP` (O(n) exact search) to `IndexIVFFlat` or `IndexHNSWFlat` for approximate nearest neighbor search with sub-millisecond latency.

2. **Embedding caching** — Cache query embeddings for repeated questions using an LRU cache to avoid redundant API calls in live mode.

3. **Async embedding calls** — Use `asyncio` with the OpenAI async client to parallelize embedding and generation calls, reducing total latency.

4. **Chunk size tuning** — Experiment with smaller chunks (400-600 tokens) for more precise retrieval, or larger chunks (1000-1200 tokens) for more context per retrieval.

5. **Re-ranking** — Add a cross-encoder re-ranking step after initial FAISS retrieval to improve relevance ordering before generation.

6. **Response caching** — Cache full responses for frequently asked questions with a TTL-based cache to reduce both latency and cost.

7. **Streaming responses** — Use Server-Sent Events (SSE) for live mode to stream LLM output tokens as they are generated, improving perceived latency.
