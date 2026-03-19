"""Tests for document ingestion and chunking."""

import os

import pytest

from app.rag.ingest import (
    chunk_document,
    count_tokens,
    ingest_documents,
    load_documents,
    parse_frontmatter,
)


class TestParseFrontmatter:
    def test_valid_frontmatter(self):
        content = '---\ntitle: "Test"\ndocument_id: "DOC-001"\n---\n\nBody text here.'
        fm, body = parse_frontmatter(content)
        assert fm["title"] == "Test"
        assert fm["document_id"] == "DOC-001"
        assert "Body text here." in body

    def test_no_frontmatter(self):
        content = "Just a plain document with no frontmatter."
        fm, body = parse_frontmatter(content)
        assert fm == {}
        assert body == content


class TestCountTokens:
    def test_english_text(self):
        tokens = count_tokens("Hello, world!")
        assert tokens > 0

    def test_arabic_text(self):
        tokens = count_tokens("مرحباً بالعالم")
        assert tokens > 0

    def test_empty_string(self):
        assert count_tokens("") == 0


class TestLoadDocuments:
    def test_loads_all_documents(self, documents_dir):
        docs = load_documents(documents_dir)
        assert len(docs) == 5

    def test_metadata_fields(self, documents_dir):
        docs = load_documents(documents_dir)
        for metadata, body in docs:
            assert metadata.title
            assert metadata.document_id
            assert metadata.source_file.endswith(".md")
            assert len(body) > 100  # non-trivial content

    def test_invalid_directory(self):
        with pytest.raises(FileNotFoundError):
            load_documents("/nonexistent/path")


class TestChunkDocument:
    def test_chunks_have_metadata(self, documents_dir):
        docs = load_documents(documents_dir)
        metadata, body = docs[0]
        chunks = chunk_document(metadata, body)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.document_id == metadata.document_id
            assert chunk.document_title == metadata.title
            assert chunk.source_file == metadata.source_file
            assert chunk.text

    def test_chunk_size_limit(self, documents_dir):
        docs = load_documents(documents_dir)
        metadata, body = docs[0]
        chunk_size = 400
        chunks = chunk_document(metadata, body, chunk_size=chunk_size)
        for chunk in chunks:
            tokens = count_tokens(chunk.text)
            # Allow some overflow due to splitting granularity
            assert tokens <= chunk_size * 1.5, (
                f"Chunk has {tokens} tokens, expected <= {chunk_size * 1.5}"
            )

    def test_chunk_indices_sequential(self, documents_dir):
        docs = load_documents(documents_dir)
        metadata, body = docs[0]
        chunks = chunk_document(metadata, body)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)


class TestIngestDocuments:
    def test_full_ingestion(self, documents_dir):
        all_chunks = ingest_documents(documents_dir)
        assert len(all_chunks) > 10  # expect at least 10 chunks across 5 docs

    def test_all_documents_represented(self, documents_dir):
        all_chunks = ingest_documents(documents_dir)
        doc_ids = set(c.document_id for c in all_chunks)
        assert len(doc_ids) == 5
