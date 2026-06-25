"""Load and chunk documents from a directory.

Supports PDF (``.pdf``), Markdown (``.md``) and plain text (``.txt``) files.
Each source file is split into overlapping chunks suitable for embedding.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}


def _load_file(path: Path) -> list[Document]:
    """Load a single file into one or more LangChain ``Document`` objects."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif suffix in {".md", ".markdown", ".txt"}:
        docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {path}")

    # Normalise metadata so downstream stages have a stable `source` field.
    for doc in docs:
        doc.metadata.setdefault("source", path.name)
        doc.metadata["file_path"] = str(path)
    return docs


def load_documents(docs_dir: str | None = None) -> list[Document]:
    """Load every supported document found under ``docs_dir``.

    Args:
        docs_dir: Directory to scan. Defaults to ``settings.docs_dir``.

    Returns:
        A flat list of loaded documents (one or more per source file).
    """
    directory = Path(docs_dir or settings.docs_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Docs directory does not exist: {directory}")

    documents: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            logger.info("Loading %s", path)
            documents.extend(_load_file(path))

    logger.info("Loaded %d document(s) from %s", len(documents), directory)
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into overlapping chunks.

    Args:
        documents: Documents produced by :func:`load_documents`.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.

    Returns:
        A list of chunk documents with a ``chunk_id`` added to each metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    logger.info("Split %d document(s) into %d chunk(s)", len(documents), len(chunks))
    return chunks


def load_and_chunk(docs_dir: str | None = None) -> list[Document]:
    """Convenience helper: load then chunk in one call."""
    return chunk_documents(load_documents(docs_dir))
