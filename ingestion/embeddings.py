"""Embed chunks with sentence-transformers and persist them to ChromaDB."""

from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import settings

logger = logging.getLogger(__name__)

# Module-level cache so the (relatively heavy) embedding model loads once.
_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached sentence-transformers embedding function."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return _embeddings


def get_vector_store() -> Chroma:
    """Return a Chroma vector store backed by the configured directory."""
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )


def build_embeddings(chunks: list[Document]) -> Chroma:
    """Embed ``chunks`` and add them to the Chroma collection.

    Args:
        chunks: Chunk documents produced by the loader.

    Returns:
        The populated :class:`~langchain_chroma.Chroma` vector store.
    """
    store = get_vector_store()
    if chunks:
        store.add_documents(chunks)
        logger.info(
            "Embedded %d chunk(s) into Chroma collection '%s'",
            len(chunks),
            settings.chroma_collection,
        )
    return store
