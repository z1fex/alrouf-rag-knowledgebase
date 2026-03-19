"""FAISS vector store for indexing and searching document chunk embeddings.

Uses IndexFlatIP (inner product on L2-normalized vectors = cosine similarity).
Stores metadata alongside the index for retrieval of chunk text and source info.
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

from app.models.schemas import ChunkMetadata


class VectorStore:
    """FAISS-based vector store for chunk embeddings.

    Attributes:
        index_dir: Directory where the FAISS index and metadata are stored.
        index: The FAISS index object.
        metadata: List of ChunkMetadata aligned by FAISS index position.
    """

    INDEX_FILENAME = "index.faiss"
    METADATA_FILENAME = "metadata.pkl"

    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[ChunkMetadata] = []

    def build(
        self, embeddings: NDArray[np.float32], chunks: list[ChunkMetadata]
    ) -> None:
        """Build a new FAISS index from embeddings and metadata.

        Args:
            embeddings: 2D array of shape (n_chunks, embedding_dim).
            chunks: List of ChunkMetadata, one per embedding row.
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedding count ({len(embeddings)}) != chunk count ({len(chunks)})"
            )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.metadata = chunks

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("No index to save. Call build() first.")

        os.makedirs(self.index_dir, exist_ok=True)

        index_path = os.path.join(self.index_dir, self.INDEX_FILENAME)
        metadata_path = os.path.join(self.index_dir, self.METADATA_FILENAME)

        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self) -> None:
        """Load a FAISS index and metadata from disk.

        Raises:
            FileNotFoundError: If the index files don't exist.
        """
        index_path = os.path.join(self.index_dir, self.INDEX_FILENAME)
        metadata_path = os.path.join(self.index_dir, self.METADATA_FILENAME)

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Index not found at {self.index_dir}. "
                "Run: python -m scripts.build_index"
            )

        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(
        self, query_embedding: NDArray[np.float32], top_k: int = 3
    ) -> list[tuple[ChunkMetadata, float]]:
        """Search the index for the most similar chunks.

        Args:
            query_embedding: 2D array of shape (1, embedding_dim).
            top_k: Number of results to return.

        Returns:
            List of (ChunkMetadata, similarity_score) tuples, sorted by
            descending similarity.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() or build() first.")

        # Clamp top_k to the number of indexed vectors
        top_k = min(top_k, self.index.ntotal)
        if top_k == 0:
            return []

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            chunk = self.metadata[idx]
            # Clamp score to [0, 1] range for cosine similarity
            clamped_score = float(max(0.0, min(1.0, score)))
            results.append((chunk, clamped_score))

        return results

    @property
    def is_loaded(self) -> bool:
        """Check if the index is loaded and ready for search."""
        return self.index is not None and len(self.metadata) > 0

    @property
    def total_chunks(self) -> int:
        """Return the total number of indexed chunks."""
        if self.index is None:
            return 0
        return self.index.ntotal

    @property
    def dimension(self) -> int:
        """Return the embedding dimension of the index."""
        if self.index is None:
            return 0
        return self.index.d

    def index_exists(self) -> bool:
        """Check if a saved index exists on disk."""
        index_path = os.path.join(self.index_dir, self.INDEX_FILENAME)
        metadata_path = os.path.join(self.index_dir, self.METADATA_FILENAME)
        return os.path.exists(index_path) and os.path.exists(metadata_path)
