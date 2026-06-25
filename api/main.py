"""FastAPI service for the GraphRAG Docs Assistant.

Endpoints
---------
- ``GET  /health``   — liveness probe.
- ``POST /chat``     — answer a question with hybrid retrieval + generation.
- ``POST /ingest``   — load/chunk/embed docs and build the knowledge graph.
- ``POST /feedback`` — record thumbs up/down on an answer.
- ``GET  /``         — serve the minimal chat frontend.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import settings
from db import store
from generation import llm
from ingestion import embeddings, graph_builder, loader
from retrieval import hybrid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GraphRAG Docs Assistant", version="1.0.0")

_FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    question: str = Field(..., description="User question")
    session_id: str | None = Field(None, description="Existing chat session UUID")
    k: int = Field(4, ge=1, le=20, description="Vector chunks to retrieve")


class ChatResponse(BaseModel):
    session_id: str
    message_id: int
    answer: str
    citations: list[str]
    graph_facts: list[str]


class IngestRequest(BaseModel):
    docs_dir: str | None = Field(None, description="Override DOCS_DIR for this run")


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    nodes: int
    relationships: int


class FeedbackRequest(BaseModel):
    message_id: int
    rating: int = Field(..., description="1 for thumbs up, -1 for thumbs down")
    comment: str | None = None


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
@app.on_event("startup")
def _startup() -> None:
    """Initialise the PostgreSQL schema (best-effort)."""
    try:
        store.init_schema()
    except Exception as exc:  # pragma: no cover - infra may be down at boot
        logger.warning("Could not initialise PostgreSQL schema at startup: %s", exc)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe reporting the active LLM provider/model."""
    model = settings.anthropic_model if settings.llm_provider == "anthropic" else settings.ollama_model
    return {"status": "ok", "provider": settings.llm_provider, "model": model}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Answer a question using hybrid retrieval and the configured LLM."""
    session_id = req.session_id or store.create_session(title=req.question[:80])
    store.add_message(session_id, "user", req.question)

    result = hybrid.retrieve(req.question, k=req.k)
    answer_text = llm.answer(req.question, result.vector_context, result.graph_context)

    citations = result.citations
    message_id = store.add_message(session_id, "assistant", answer_text, citations)

    return ChatResponse(
        session_id=session_id,
        message_id=message_id,
        answer=answer_text,
        citations=citations,
        graph_facts=[t.render() for t in result.triples],
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """Run the full ingestion pipeline over the docs directory."""
    documents = loader.load_documents(req.docs_dir)
    if not documents:
        raise HTTPException(status_code=400, detail="No documents found to ingest.")

    chunks = loader.chunk_documents(documents)
    embeddings.build_embeddings(chunks)
    stats = graph_builder.build_graph(chunks)

    # Persist per-source metadata.
    sources = {d.metadata.get("source", "unknown") for d in documents}
    for source in sources:
        src_chunks = [c for c in chunks if c.metadata.get("source") == source]
        file_path = next(
            (d.metadata.get("file_path") for d in documents if d.metadata.get("source") == source),
            None,
        )
        store.record_document(source, file_path, len(src_chunks), stats.nodes, stats.relationships)

    return IngestResponse(
        documents=len(sources),
        chunks=len(chunks),
        nodes=stats.nodes,
        relationships=stats.relationships,
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict[str, int | str]:
    """Record thumbs up/down feedback for a generated answer."""
    if req.rating not in (-1, 1):
        raise HTTPException(status_code=400, detail="rating must be 1 or -1")
    feedback_id = store.add_feedback(req.message_id, req.rating, req.comment)
    return {"status": "ok", "feedback_id": feedback_id}


@app.get("/")
def index() -> FileResponse:
    """Serve the minimal chat frontend."""
    if not _FRONTEND.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(_FRONTEND)
