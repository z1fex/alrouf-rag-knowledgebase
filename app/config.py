"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings read from environment variables or .env file."""

    # Mode
    use_mock_llm: bool = Field(default=True, alias="USE_MOCK_LLM")

    # OpenAI (only needed when USE_MOCK_LLM=false)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")

    # Retrieval
    top_k: int = Field(default=3, alias="TOP_K")
    similarity_threshold: float = Field(default=0.3, alias="SIMILARITY_THRESHOLD")

    # Paths
    documents_dir: str = Field(default="data/documents", alias="DOCUMENTS_DIR")
    vectorstore_dir: str = Field(default="vectorstore", alias="VECTORSTORE_DIR")
    mocks_dir: str = Field(default="mocks", alias="MOCKS_DIR")

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    @property
    def index_dir(self) -> str:
        """Return the appropriate index directory based on mode."""
        mode = "mock" if self.use_mock_llm else "live"
        return f"{self.vectorstore_dir}/{mode}"


settings = Settings()
