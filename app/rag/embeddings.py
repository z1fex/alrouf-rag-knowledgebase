"""Embedding generation for the RAG pipeline.

Provides two implementations:
- MockEmbedder: Uses TF-IDF + TruncatedSVD (scikit-learn) for offline/mock mode.
  Handles Arabic via character n-gram analysis — no external tokenizer needed.
- OpenAIEmbedder: Calls the OpenAI embeddings API for live mode.
"""

import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""

    @abstractmethod
    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            2D numpy array of shape (len(texts), embedding_dim).
        """
        ...

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding vector dimension."""
        ...


class MockEmbedder(EmbeddingService):
    """TF-IDF + TruncatedSVD embedder for mock/offline mode.

    Uses character n-grams (char_wb, 2-4) to handle Arabic text without
    requiring a specialized tokenizer. Produces dense vectors via SVD
    that can be indexed in FAISS for semantic retrieval.
    """

    def __init__(self, n_components: int = 384, mocks_dir: str = "mocks"):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.n_components = n_components
        self.mocks_dir = mocks_dir
        self.model_path = os.path.join(mocks_dir, "tfidf_model.pkl")
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self._fitted = False

        # Try loading a pre-fitted model
        self._load_model()

    def _load_model(self) -> None:
        """Load a previously fitted TF-IDF + SVD model from disk."""
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.svd = data["svd"]
            self._fitted = True

    def fit(self, texts: list[str]) -> None:
        """Fit the TF-IDF vectorizer and SVD on a corpus of texts.

        Args:
            texts: Corpus of text strings to fit on.
        """
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=10000,
            sublinear_tf=True,
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        # SVD components must not exceed matrix dimensions
        actual_components = min(self.n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=actual_components, random_state=42)
        self.svd.fit(tfidf_matrix)
        self.n_components = actual_components
        self._fitted = True

        # Save model to disk
        self._save_model()

    def _save_model(self) -> None:
        """Persist the fitted model to disk."""
        os.makedirs(self.mocks_dir, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "svd": self.svd}, f)

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings using the fitted TF-IDF + SVD pipeline.

        Args:
            texts: List of text strings to embed.

        Returns:
            Normalized embedding vectors of shape (len(texts), n_components).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._fitted or self.vectorizer is None or self.svd is None:
            raise RuntimeError(
                "MockEmbedder is not fitted. Run fit() first or build the index "
                "with: python -m scripts.build_index"
            )

        tfidf_matrix = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf_matrix).astype(np.float32)

        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid division by zero
        return dense / norms

    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self.n_components


class OpenAIEmbedder(EmbeddingService):
    """OpenAI API-based embedder for live mode.

    Uses the text-embedding-3-small model (1536 dimensions) by default.
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._dimension = 1536  # text-embedding-3-small default

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings by calling the OpenAI API.

        Args:
            texts: List of text strings to embed.

        Returns:
            Normalized embedding vectors of shape (len(texts), 1536).
        """
        # OpenAI API accepts batches up to ~2048 inputs
        response = self.client.embeddings.create(input=texts, model=self.model)
        embeddings = [item.embedding for item in response.data]
        result = np.array(embeddings, dtype=np.float32)

        # Normalize
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return result / norms

    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension


def get_embedding_service(
    use_mock: bool = True,
    api_key: str = "",
    model: str = "text-embedding-3-small",
    mocks_dir: str = "mocks",
) -> EmbeddingService:
    """Factory function to create the appropriate embedding service.

    Args:
        use_mock: If True, use TF-IDF mock embedder; otherwise use OpenAI.
        api_key: OpenAI API key (required if use_mock is False).
        model: OpenAI embedding model name.
        mocks_dir: Directory for mock model files.

    Returns:
        An EmbeddingService instance.
    """
    if use_mock:
        return MockEmbedder(mocks_dir=mocks_dir)
    else:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when USE_MOCK_LLM=false")
        return OpenAIEmbedder(api_key=api_key, model=model)
