"""Document loading and chunking for the RAG pipeline.

Loads markdown documents from the data directory, parses YAML frontmatter
for metadata, and splits content into overlapping chunks sized by token count.
"""

import os
import re
from pathlib import Path

import tiktoken
import yaml

from app.models.schemas import ChunkMetadata, DocumentMetadata


# tiktoken encoding for OpenAI models
ENCODING = tiktoken.get_encoding("cl100k_base")

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 800  # tokens
DEFAULT_CHUNK_OVERLAP = 100  # tokens


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string using cl100k_base encoding."""
    return len(ENCODING.encode(text))


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from a markdown document.

    Args:
        content: Raw markdown string with optional YAML frontmatter.

    Returns:
        Tuple of (frontmatter dict, body text without frontmatter).
    """
    pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(pattern, content, re.DOTALL)
    if match:
        frontmatter = yaml.safe_load(match.group(1)) or {}
        body = content[match.end():]
        return frontmatter, body
    return {}, content


def load_documents(documents_dir: str) -> list[tuple[DocumentMetadata, str]]:
    """Load all markdown documents from the specified directory.

    Args:
        documents_dir: Path to the directory containing .md files.

    Returns:
        List of (DocumentMetadata, body_text) tuples.
    """
    docs = []
    doc_dir = Path(documents_dir)

    if not doc_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    for filepath in sorted(doc_dir.glob("*.md")):
        content = filepath.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(content)

        metadata = DocumentMetadata(
            title=frontmatter.get("title", filepath.stem),
            title_ar=frontmatter.get("title_ar", ""),
            document_id=frontmatter.get("document_id", filepath.stem),
            source_file=filepath.name,
        )
        docs.append((metadata, body))

    if not docs:
        raise ValueError(f"No markdown documents found in {documents_dir}")

    return docs


def _split_by_separators(text: str, separators: list[str]) -> list[str]:
    """Recursively split text by a hierarchy of separators.

    Tries the first separator; if resulting pieces are still too large
    they'll be further split in the chunking step.
    """
    if not separators:
        return [text]

    sep = separators[0]
    remaining_seps = separators[1:]

    parts = text.split(sep)
    # Re-attach separator to each part (except the first)
    result = []
    for i, part in enumerate(parts):
        if i > 0:
            part = sep + part
        if part.strip():
            result.append(part)

    # If no split occurred, try next separator
    if len(result) <= 1 and remaining_seps:
        return _split_by_separators(text, remaining_seps)

    return result


def chunk_document(
    metadata: DocumentMetadata,
    body: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkMetadata]:
    """Split a document body into overlapping chunks with metadata.

    Uses a hierarchical splitting strategy:
    1. Split on '## ' (h2 headers)
    2. Split on '### ' (h3 headers)
    3. Split on double newlines (paragraphs)
    4. Split on single newlines
    5. Split on sentence boundaries

    Args:
        metadata: Document metadata from frontmatter.
        body: The document body text.
        chunk_size: Maximum tokens per chunk.
        chunk_overlap: Number of overlapping tokens between consecutive chunks.

    Returns:
        List of ChunkMetadata objects with text content.
    """
    separators = ["\n## ", "\n### ", "\n\n", "\n", ". "]
    sections = _split_by_separators(body, separators)

    # Merge small sections and split large ones into token-bounded chunks
    chunks_text: list[str] = []
    current_chunk = ""

    for section in sections:
        section_tokens = count_tokens(section)

        # If a single section exceeds chunk_size, force-split it
        if section_tokens > chunk_size:
            # Flush current chunk first
            if current_chunk.strip():
                chunks_text.append(current_chunk.strip())
                current_chunk = ""

            # Split large section by smaller separators
            words = section.split(" ")
            piece = ""
            for word in words:
                candidate = piece + " " + word if piece else word
                if count_tokens(candidate) > chunk_size and piece:
                    chunks_text.append(piece.strip())
                    # Keep overlap
                    overlap_words = piece.strip().split(" ")
                    overlap_token_count = 0
                    overlap_start = len(overlap_words)
                    for j in range(len(overlap_words) - 1, -1, -1):
                        overlap_token_count += count_tokens(overlap_words[j] + " ")
                        if overlap_token_count >= chunk_overlap:
                            overlap_start = j
                            break
                    piece = " ".join(overlap_words[overlap_start:]) + " " + word
                else:
                    piece = candidate
            if piece.strip():
                chunks_text.append(piece.strip())
            continue

        # Check if adding this section exceeds chunk_size
        combined = current_chunk + "\n" + section if current_chunk else section
        if count_tokens(combined) > chunk_size and current_chunk.strip():
            chunks_text.append(current_chunk.strip())
            # Create overlap from end of current chunk
            overlap_text = _get_overlap_text(current_chunk, chunk_overlap)
            current_chunk = overlap_text + "\n" + section if overlap_text else section
        else:
            current_chunk = combined

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks_text.append(current_chunk.strip())

    # Build ChunkMetadata objects
    total = len(chunks_text)
    return [
        ChunkMetadata(
            document_id=metadata.document_id,
            document_title=metadata.title,
            source_file=metadata.source_file,
            chunk_index=i,
            total_chunks=total,
            text=text,
        )
        for i, text in enumerate(chunks_text)
    ]


def _get_overlap_text(text: str, overlap_tokens: int) -> str:
    """Extract the last `overlap_tokens` worth of text for chunk overlap."""
    words = text.split(" ")
    token_count = 0
    start_idx = len(words)
    for i in range(len(words) - 1, -1, -1):
        token_count += count_tokens(words[i] + " ")
        if token_count >= overlap_tokens:
            start_idx = i
            break
    return " ".join(words[start_idx:])


def ingest_documents(
    documents_dir: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkMetadata]:
    """Full ingestion pipeline: load documents, chunk them, return all chunks.

    Args:
        documents_dir: Path to the documents directory.
        chunk_size: Maximum tokens per chunk.
        chunk_overlap: Overlap tokens between chunks.

    Returns:
        List of all ChunkMetadata across all documents.
    """
    documents = load_documents(documents_dir)
    all_chunks: list[ChunkMetadata] = []

    for metadata, body in documents:
        chunks = chunk_document(metadata, body, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    return all_chunks
