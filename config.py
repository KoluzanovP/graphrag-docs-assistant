"""Central configuration loaded from environment variables.

All modules import :data:`settings` from here so that environment handling
lives in one place. Values are read once at import time via ``python-dotenv``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    """Typed view over the process environment."""

    # LLM provider
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic"))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    # The default Anthropic model id is claude-opus-4-8.
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1"))

    # Embeddings
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # Chroma
    chroma_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", "./.chroma"))
    chroma_collection: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "graphrag_docs"))

    # Neo4j
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "neo4j"))

    # PostgreSQL
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "graphrag"))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "graphrag"))
    postgres_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "graphrag"))

    # Ingestion
    docs_dir: str = field(default_factory=lambda: os.getenv("DOCS_DIR", "./docs"))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "120")))

    @property
    def postgres_dsn(self) -> str:
        """libpq connection string for psycopg2."""
        return (
            f"host={self.postgres_host} port={self.postgres_port} "
            f"dbname={self.postgres_db} user={self.postgres_user} "
            f"password={self.postgres_password}"
        )


settings = Settings()
