"""Vector similarity search over the ChromaDB collection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document

from ingestion.embeddings import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class VectorHit:
    """A single similarity-search result."""

    content: str
    source: str
    chunk_id: int
    score: float

    def citation(self) -> str:
        return f"[source: {self.source}#{self.chunk_id}]"


def search(query: str, k: int = 4) -> list[VectorHit]:
    """Return the top-``k`` chunks most similar to ``query``.

    Args:
        query: Natural-language query.
        k: Number of chunks to retrieve.

    Returns:
        A list of :class:`VectorHit` ordered by descending relevance.
    """
    store = get_vector_store()
    results: list[tuple[Document, float]] = store.similarity_search_with_score(query, k=k)
    hits: list[VectorHit] = []
    for doc, score in results:
        hits.append(
            VectorHit(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                chunk_id=int(doc.metadata.get("chunk_id", -1)),
                score=float(score),
            )
        )
    logger.info("Vector search for %r returned %d hit(s)", query, len(hits))
    return hits


def format_context(hits: list[VectorHit]) -> str:
    """Render vector hits as a citation-prefixed context block."""
    return "\n\n".join(f"{hit.citation()} {hit.content}" for hit in hits)
