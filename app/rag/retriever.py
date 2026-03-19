"""Retriever that combines embedding and vector search to find relevant chunks.

Embeds the user's query, searches the FAISS index, and returns ranked
chunks with metadata and similarity scores.
"""

from app.models.schemas import ChunkMetadata
from app.rag.embeddings import EmbeddingService
from app.rag.vectorstore import VectorStore


class Retriever:
    """Orchestrates query embedding and vector search.

    Attributes:
        embedding_service: Service for generating query embeddings.
        vector_store: FAISS vector store with indexed document chunks.
        top_k: Default number of results to retrieve.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        top_k: int = 3,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[tuple[ChunkMetadata, float]]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: The user's question text.
            top_k: Number of results to return. Uses default if not specified.

        Returns:
            List of (ChunkMetadata, similarity_score) tuples sorted by
            descending relevance. Each ChunkMetadata includes the chunk
            text in its `text` field.
        """
        k = top_k if top_k is not None else self.top_k

        # Embed the query
        query_embedding = self.embedding_service.embed([query])

        # Search the vector store
        results = self.vector_store.search(query_embedding, top_k=k)

        return results
