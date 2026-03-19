"""FastAPI application for the Alrouf RAG Knowledge Base.

Provides endpoints for querying the knowledge base, health checks,
and serving the web UI.
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models.schemas import (
    Citation,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from app.rag.embeddings import MockEmbedder, get_embedding_service
from app.rag.generator import get_generator
from app.rag.ingest import ingest_documents
from app.rag.retriever import Retriever
from app.rag.vectorstore import VectorStore
from app.utils.scope_checker import get_refusal_message, is_in_scope

# Global instances initialized at startup
_retriever: Retriever | None = None
_generator = None
_vector_store: VectorStore | None = None


def _build_index() -> None:
    """Build the vector index from source documents."""
    global _retriever, _generator, _vector_store

    embedder = get_embedding_service(
        use_mock=settings.use_mock_llm,
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        mocks_dir=settings.mocks_dir,
    )

    # Ingest and chunk documents
    chunks = ingest_documents(settings.documents_dir)

    # Fit mock embedder if needed
    if isinstance(embedder, MockEmbedder):
        corpus = [chunk.text for chunk in chunks]
        embedder.fit(corpus)

    # Generate embeddings and build index
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed(texts)

    _vector_store = VectorStore(index_dir=settings.index_dir)
    _vector_store.build(embeddings, chunks)
    _vector_store.save()

    _retriever = Retriever(embedder, _vector_store, top_k=settings.top_k)
    _generator = get_generator(
        use_mock=settings.use_mock_llm,
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        mocks_dir=settings.mocks_dir,
    )


def _load_index() -> None:
    """Load an existing vector index from disk."""
    global _retriever, _generator, _vector_store

    embedder = get_embedding_service(
        use_mock=settings.use_mock_llm,
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        mocks_dir=settings.mocks_dir,
    )

    _vector_store = VectorStore(index_dir=settings.index_dir)
    _vector_store.load()

    _retriever = Retriever(embedder, _vector_store, top_k=settings.top_k)
    _generator = get_generator(
        use_mock=settings.use_mock_llm,
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        mocks_dir=settings.mocks_dir,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load or build the index on startup."""
    vs = VectorStore(index_dir=settings.index_dir)
    if vs.index_exists():
        print(f"Loading existing index from {settings.index_dir}...")
        _load_index()
    else:
        print("No existing index found. Building from documents...")
        _build_index()
    print(f"Index ready. Mode: {'MOCK' if settings.use_mock_llm else 'LIVE'}")
    yield


app = FastAPI(
    title="Alrouf RAG Knowledge Base",
    description="Bilingual (Arabic/English) Q&A system for Alrouf Lighting Technology",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Process a user query and return an answer with citations.

    Args:
        request: QueryRequest with query text and language preference.

    Returns:
        QueryResponse with answer, citations, scope status, and latency.
    """
    if _retriever is None or _generator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    start_time = time.time()

    # Retrieve relevant chunks
    results = _retriever.retrieve(request.query)

    # Check scope
    in_scope = is_in_scope(
        request.query,
        results,
        similarity_threshold=settings.similarity_threshold,
    )

    if not in_scope:
        latency_ms = (time.time() - start_time) * 1000
        return QueryResponse(
            query=request.query,
            answer=get_refusal_message(request.language),
            citations=[],
            language=request.language,
            is_in_scope=False,
            latency_ms=round(latency_ms, 2),
        )

    # Generate answer
    answer, citations = _generator.generate(
        request.query, results, request.language
    )

    latency_ms = (time.time() - start_time) * 1000

    return QueryResponse(
        query=request.query,
        answer=answer,
        citations=citations,
        language=request.language,
        is_in_scope=True,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health and index statistics."""
    if _vector_store is None or not _vector_store.is_loaded:
        return HealthResponse(
            status="not_ready",
            mode="mock" if settings.use_mock_llm else "live",
            documents_indexed=0,
            total_chunks=0,
            index_dimensions=0,
        )

    # Count unique documents
    doc_ids = set()
    for chunk in _vector_store.metadata:
        doc_ids.add(chunk.document_id)

    return HealthResponse(
        status="healthy",
        mode="mock" if settings.use_mock_llm else "live",
        documents_indexed=len(doc_ids),
        total_chunks=_vector_store.total_chunks,
        index_dimensions=_vector_store.dimension,
    )


@app.post("/api/reindex")
async def reindex():
    """Rebuild the vector index from source documents."""
    try:
        _build_index()
        return {"status": "ok", "message": "Index rebuilt successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI."""
    html_path = static_dir / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
