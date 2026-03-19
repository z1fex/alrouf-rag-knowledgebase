"""Tests for answer generation."""

import pytest

from app.models.schemas import ChunkMetadata
from app.rag.generator import MockGenerator


@pytest.fixture
def mock_generator(mocks_dir):
    return MockGenerator(mocks_dir=mocks_dir)


@pytest.fixture
def sample_chunks():
    """Create sample retrieved chunks for testing."""
    return [
        (
            ChunkMetadata(
                document_id="DOC-001",
                document_title="Alrouf Product Catalog 2024",
                source_file="product_catalog_2024.md",
                chunk_index=2,
                total_chunks=10,
                text="The ALR-SL-90W has IP66 rating and IK09 impact protection.",
            ),
            0.85,
        ),
        (
            ChunkMetadata(
                document_id="DOC-005",
                document_title="Technical Specifications — Street Lighting Series",
                source_file="technical_specs_streetlighting.md",
                chunk_index=1,
                total_chunks=8,
                text="ALR-SL-90W: IP66, IK09, wind resistance up to 150 km/h.",
            ),
            0.72,
        ),
    ]


class TestMockGenerator:
    def test_pattern_match_ip_rating(self, mock_generator, sample_chunks):
        answer, citations = mock_generator.generate(
            "What is the IP rating of the ALR-SL-90W?",
            sample_chunks,
            language="en",
        )
        assert "IP66" in answer
        assert len(citations) == 2

    def test_pattern_match_warranty(self, mock_generator, sample_chunks):
        answer, citations = mock_generator.generate(
            "What is the warranty period?",
            sample_chunks,
            language="en",
        )
        assert "5 years" in answer or "warranty" in answer.lower()

    def test_arabic_response(self, mock_generator, sample_chunks):
        answer, citations = mock_generator.generate(
            "ما هي شهادات الجودة لشركة الروف؟",
            sample_chunks,
            language="ar",
        )
        # Should return Arabic text
        assert any(c in answer for c in "ابتثجحخدذرزسشصضطظعغفقكلمنهوي")

    def test_citations_structure(self, mock_generator, sample_chunks):
        _, citations = mock_generator.generate(
            "What is the IP rating?",
            sample_chunks,
            language="en",
        )
        for citation in citations:
            assert citation.document_title
            assert citation.document_id
            assert citation.source_file
            assert 0.0 <= citation.relevance_score <= 1.0

    def test_fallback_response(self, mock_generator, sample_chunks):
        """Test that a non-matching query still generates an answer from chunks."""
        answer, citations = mock_generator.generate(
            "Tell me something very specific about nothing in particular",
            sample_chunks,
            language="en",
        )
        assert len(answer) > 0
        assert len(citations) == 2

    def test_empty_chunks(self, mock_generator):
        answer, citations = mock_generator.generate(
            "What is the IP rating?",
            [],
            language="en",
        )
        assert len(answer) > 0  # Should return out-of-scope message
        assert len(citations) == 0
