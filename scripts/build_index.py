"""CLI script to ingest documents, generate embeddings, build FAISS index.

Usage:
    python -m scripts.build_index

This script:
1. Loads all markdown documents from data/documents/
2. Chunks them with token-aware splitting
3. Generates embeddings (mock TF-IDF or OpenAI depending on config)
4. Builds and saves a FAISS index to vectorstore/{mock|live}/
"""

import os
import sys
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.rag.embeddings import MockEmbedder, get_embedding_service
from app.rag.ingest import ingest_documents
from app.rag.vectorstore import VectorStore


def main() -> None:
    """Build the vector index from source documents."""
    print("=" * 60)
    print("Alrouf RAG Knowledge Base — Index Builder")
    print("=" * 60)
    print(f"Mode: {'MOCK' if settings.use_mock_llm else 'LIVE'}")
    print(f"Documents dir: {settings.documents_dir}")
    print(f"Index dir: {settings.index_dir}")
    print()

    # Step 1: Ingest documents
    print("[1/4] Loading and chunking documents...")
    start = time.time()
    chunks = ingest_documents(settings.documents_dir)
    ingest_time = time.time() - start
    print(f"  -> {len(chunks)} chunks from documents")
    print(f"  -> Ingest time: {ingest_time:.2f}s")
    print()

    # Step 2: Create embedding service
    print("[2/4] Initializing embedding service...")
    embedder = get_embedding_service(
        use_mock=settings.use_mock_llm,
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        mocks_dir=settings.mocks_dir,
    )

    # For mock mode, we need to fit the TF-IDF model on the corpus
    if isinstance(embedder, MockEmbedder):
        print("  -> Fitting TF-IDF + SVD model on corpus...")
        corpus = [chunk.text for chunk in chunks]
        embedder.fit(corpus)
        print(f"  -> Model saved to {settings.mocks_dir}/tfidf_model.pkl")

    print()

    # Step 3: Generate embeddings
    print("[3/4] Generating embeddings...")
    start = time.time()
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed(texts)
    embed_time = time.time() - start
    print(f"  -> Embedding shape: {embeddings.shape}")
    print(f"  -> Embedding time: {embed_time:.2f}s")
    print()

    # Step 4: Build and save FAISS index
    print("[4/4] Building FAISS index...")
    start = time.time()
    store = VectorStore(index_dir=settings.index_dir)
    store.build(embeddings, chunks)
    store.save()
    index_time = time.time() - start
    print(f"  -> Index saved to {settings.index_dir}/")
    print(f"  -> Total vectors: {store.total_chunks}")
    print(f"  -> Dimensions: {store.dimension}")
    print(f"  -> Index build time: {index_time:.2f}s")
    print()

    # Summary
    print("=" * 60)
    print("Index build complete!")
    print(f"  Total time: {ingest_time + embed_time + index_time:.2f}s")
    print(f"  Chunks indexed: {store.total_chunks}")
    print(f"  Vector dimensions: {store.dimension}")
    print("=" * 60)


if __name__ == "__main__":
    main()
