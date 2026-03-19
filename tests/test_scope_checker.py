"""Tests for out-of-scope detection."""

import pytest

from app.models.schemas import ChunkMetadata
from app.utils.scope_checker import get_refusal_message, is_in_scope


def _make_chunk(text: str, score: float):
    """Helper to create a (ChunkMetadata, score) tuple."""
    return (
        ChunkMetadata(
            document_id="DOC-001",
            document_title="Test Document",
            source_file="test.md",
            chunk_index=0,
            total_chunks=1,
            text=text,
        ),
        score,
    )


class TestIsInScope:
    def test_high_similarity_in_scope(self):
        results = [_make_chunk("ALR-SL-90W street light specifications", 0.8)]
        assert is_in_scope("What is the ALR-SL-90W?", results) is True

    def test_low_similarity_out_of_scope(self):
        results = [_make_chunk("Some random text about nothing", 0.1)]
        assert is_in_scope("What is the weather today?", results) is False

    def test_no_results_out_of_scope(self):
        assert is_in_scope("Any question", []) is False

    def test_moderate_similarity_with_keywords(self):
        results = [_make_chunk("LED lighting specifications", 0.4)]
        assert is_in_scope("Tell me about LED lights", results) is True

    def test_moderate_similarity_without_keywords(self):
        results = [_make_chunk("Something completely unrelated to anything", 0.35)]
        assert is_in_scope("What is 2+2?", results) is False

    def test_arabic_query_with_keywords(self):
        results = [_make_chunk("شركة الروف لتقنية الإضاءة", 0.45)]
        assert is_in_scope("ما هي منتجات الروف؟", results) is True

    def test_threshold_boundary(self):
        results = [_make_chunk("Some text", 0.29)]
        assert is_in_scope("query", results, similarity_threshold=0.3) is False

        results = [_make_chunk("Some text about lighting", 0.31)]
        assert is_in_scope("query about lighting", results, similarity_threshold=0.3) is True


class TestRefusalMessage:
    def test_english_refusal(self):
        msg = get_refusal_message("en")
        assert "Alrouf" in msg
        assert "products" in msg.lower()

    def test_arabic_refusal(self):
        msg = get_refusal_message("ar")
        assert "الروف" in msg
