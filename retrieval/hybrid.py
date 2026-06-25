"""Hybrid retrieval combining vector search with graph expansion.

Pipeline:
1. Run vector similarity search to find the most relevant chunks.
2. Derive seed entities from those chunks (via the graph's entity index, using
   simple token overlap) plus any entities matched directly from the question.
3. Expand into the knowledge graph to pull in related facts for multi-hop
   questions.
4. Return both context blocks plus structured hits for the API layer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from retrieval import graph_retriever, vector_retriever
from retrieval.graph_retriever import GraphRetriever, GraphTriple
from retrieval.vector_retriever import VectorHit

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


@dataclass
class HybridResult:
    """Everything the generation layer needs to answer a question."""

    question: str
    vector_hits: list[VectorHit] = field(default_factory=list)
    triples: list[GraphTriple] = field(default_factory=list)
    vector_context: str = ""
    graph_context: str = ""

    @property
    def citations(self) -> list[str]:
        cites = [hit.citation() for hit in self.vector_hits]
        if self.triples:
            cites.append("[source: graph]")
        return cites


def _candidate_terms(question: str, hits: list[VectorHit]) -> list[str]:
    """Collect candidate entity-name tokens from the question and top hits."""
    text = question + " " + " ".join(hit.content for hit in hits[:2])
    tokens = {t.lower() for t in _TOKEN_RE.findall(text)}
    return list(tokens)


def retrieve(question: str, k: int = 4, graph_limit: int = 25) -> HybridResult:
    """Run the full hybrid retrieval pipeline for ``question``.

    Args:
        question: The user's question.
        k: Number of vector chunks to retrieve.
        graph_limit: Maximum number of graph triples to pull in.

    Returns:
        A populated :class:`HybridResult`.
    """
    hits = vector_retriever.search(question, k=k)

    seed_entities: set[str] = set()
    with GraphRetriever() as graph:
        for term in _candidate_terms(question, hits):
            for name in graph.search_entities(term, limit=3):
                seed_entities.add(name)
        triples = graph.neighbourhood(sorted(seed_entities), limit=graph_limit)

    result = HybridResult(
        question=question,
        vector_hits=hits,
        triples=triples,
        vector_context=vector_retriever.format_context(hits),
        graph_context=graph_retriever.format_context(triples),
    )
    logger.info(
        "Hybrid retrieval: %d vector hits, %d graph triples, %d seed entities",
        len(hits),
        len(triples),
        len(seed_entities),
    )
    return result
