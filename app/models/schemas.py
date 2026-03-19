"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata extracted from document frontmatter."""

    title: str
    title_ar: str = ""
    document_id: str
    source_file: str


class ChunkMetadata(BaseModel):
    """Metadata attached to each chunk after splitting."""

    document_id: str
    document_title: str
    source_file: str
    chunk_index: int
    total_chunks: int
    text: str = ""


class Citation(BaseModel):
    """A single citation referencing a source chunk."""

    document_title: str
    document_id: str
    chunk_index: int
    source_file: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class QueryRequest(BaseModel):
    """Incoming query from the user."""

    query: str = Field(min_length=1, max_length=2000)
    language: str = Field(default="en", pattern="^(en|ar)$")


class QueryResponse(BaseModel):
    """Response returned to the user."""

    query: str
    answer: str
    citations: list[Citation]
    language: str
    is_in_scope: bool
    latency_ms: float


class HealthResponse(BaseModel):
    """Response from the health endpoint."""

    status: str
    mode: str
    documents_indexed: int
    total_chunks: int
    index_dimensions: int
