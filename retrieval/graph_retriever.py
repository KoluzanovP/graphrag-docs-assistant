"""Cypher-based retrieval from the Neo4j knowledge graph.

Given a set of seed entity names (typically surfaced from vector hits or a
keyword scan of the question), this module fetches the immediate neighbourhood
of those entities — the relationships that enable multi-hop reasoning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from neo4j import GraphDatabase

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class GraphTriple:
    """A (source)-[relation]->(target) fact from the graph."""

    source: str
    relation: str
    target: str

    def render(self) -> str:
        return f"({self.source})-[:{self.relation}]->({self.target})"


class GraphRetriever:
    """Read-only Neo4j client for neighbourhood and full-graph queries."""

    def __init__(self) -> None:
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "GraphRetriever":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def neighbourhood(self, entity_names: list[str], limit: int = 25) -> list[GraphTriple]:
        """Return relationships touching any of ``entity_names`` (case-insensitive)."""
        if not entity_names:
            return []
        cypher = """
        MATCH (s:Entity)-[r]->(t:Entity)
        WHERE toLower(s.name) IN $names OR toLower(t.name) IN $names
        RETURN s.name AS source, type(r) AS relation, t.name AS target
        LIMIT $limit
        """
        names = [n.lower() for n in entity_names]
        with self._driver.session() as session:
            records = session.run(cypher, names=names, limit=limit)
            triples = [
                GraphTriple(rec["source"], rec["relation"], rec["target"]) for rec in records
            ]
        logger.info("Graph neighbourhood for %s returned %d triple(s)", entity_names, len(triples))
        return triples

    def search_entities(self, term: str, limit: int = 10) -> list[str]:
        """Return entity names whose name contains ``term`` (case-insensitive)."""
        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($term)
        RETURN e.name AS name
        LIMIT $limit
        """
        with self._driver.session() as session:
            records = session.run(cypher, term=term, limit=limit)
            return [rec["name"] for rec in records]


def format_context(triples: list[GraphTriple]) -> str:
    """Render graph triples as a single semicolon-separated context block."""
    return "; ".join(triple.render() for triple in triples)
