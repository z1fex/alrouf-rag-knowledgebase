"""Tests for the retriever and vector store."""

import os
import tempfile

import numpy as np
import pytest

from app.rag.embeddings import MockEmbedder
from app.rag.ingest import ingest_documents
from app.rag.retriever import Retriever
from app.rag.vectorstore import VectorStore


@pytest.fixture(scope="module")
def index_components(documents_dir, mocks_dir):
    """Build a mock index for testing. Shared across all tests in this module."""
    chunks = ingest_documents(documents_dir)
    embedder = MockEmbedder(mocks_dir=mocks_dir)
    corpus = [c.text for c in chunks]
    embedder.fit(corpus)
    embeddings = embedder.embed(corpus)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(index_dir=tmpdir)
        store.build(embeddings, chunks)
        store.save()

        # Reload from disk to test persistence
        store2 = VectorStore(index_dir=tmpdir)
        store2.load()

        yield embedder, store2, chunks


class TestVectorStore:
    def test_index_loaded(self, index_components):
        _, store, chunks = index_components
        assert store.is_loaded
        assert store.total_chunks == len(chunks)
        assert store.dimension > 0

    def test_search_returns_results(self, index_components):
        embedder, store, _ = index_components
        query_emb = embedder.embed(["street light specifications"])
        results = store.search(query_emb, top_k=3)
        assert len(results) == 3
        # Scores should be descending
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_scores_in_range(self, index_components):
        embedder, store, _ = index_components
        query_emb = embedder.embed(["ALR-SL-90W IP rating"])
        results = store.search(query_emb, top_k=3)
        for _, score in results:
            assert 0.0 <= score <= 1.0


class TestRetriever:
    def test_retrieve_returns_chunks(self, index_components):
        embedder, store, _ = index_components
        retriever = Retriever(embedder, store, top_k=3)
        results = retriever.retrieve("What is the IP rating of ALR-SL-90W?")
        assert len(results) == 3
        for chunk, score in results:
            assert chunk.text
            assert chunk.document_id
            assert 0.0 <= score <= 1.0

    def test_retrieve_top_k(self, index_components):
        embedder, store, _ = index_components
        retriever = Retriever(embedder, store, top_k=5)
        results = retriever.retrieve("warranty period")
        assert len(results) == 5

    def test_retrieve_arabic_query(self, index_components):
        embedder, store, _ = index_components
        retriever = Retriever(embedder, store, top_k=3)
        results = retriever.retrieve("ما هي شهادات الجودة لشركة الروف؟")
        assert len(results) == 3
        # Should return results (not empty)
        for chunk, score in results:
            assert chunk.text
