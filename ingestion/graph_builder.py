"""LLM-based entity/relation extraction into a Neo4j knowledge graph.

For each chunk the LLM is asked to return a small JSON object describing the
entities it mentions and the relationships between them. Those are then merged
into Neo4j as ``(:Entity)-[:REL]->(:Entity)`` nodes/edges, with each entity
linked back to the chunk it came from via ``(:Chunk)-[:MENTIONS]->(:Entity)``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from langchain_core.documents import Document
from neo4j import GraphDatabase

from config import settings
from generation.llm import get_llm

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """You are an information-extraction engine building a knowledge graph.
From the text below, extract the key entities and the relationships between them.

Return ONLY valid JSON with this exact shape (no prose, no markdown fences):
{{
  "entities": [{{"name": "string", "type": "string"}}],
  "relations": [{{"source": "string", "relation": "string", "target": "string"}}]
}}

Rules:
- "type" is a short label such as Person, Organization, Concept, Technology, Product.
- "relation" is an UPPER_SNAKE_CASE verb phrase, e.g. WORKS_FOR, PART_OF, USES.
- Only include relations whose source and target both appear in "entities".
- Keep entity names canonical and concise.

TEXT:
{text}
"""


@dataclass
class GraphStats:
    """Counts returned after building the graph."""

    nodes: int = 0
    relationships: int = 0
    chunks: int = 0


def _parse_json_block(raw: str) -> dict:
    """Best-effort parse of an LLM response that should contain JSON."""
    # Strip code fences if the model added them despite instructions.
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return {"entities": [], "relations": []}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("Failed to parse extraction JSON: %s", raw[:200])
        return {"entities": [], "relations": []}
    data.setdefault("entities", [])
    data.setdefault("relations", [])
    return data


def extract_graph(text: str) -> dict:
    """Run the LLM extraction prompt over a single chunk of text."""
    llm = get_llm()
    response = llm.invoke(_EXTRACTION_PROMPT.format(text=text))
    content = getattr(response, "content", response)
    return _parse_json_block(content if isinstance(content, str) else str(content))


class Neo4jGraphWriter:
    """Thin wrapper around the Neo4j driver for MERGE-based graph writes."""

    def __init__(self) -> None:
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jGraphWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def write_chunk_graph(self, chunk_id: int, source: str, extraction: dict) -> tuple[int, int]:
        """MERGE the entities/relations from one chunk. Returns (nodes, rels)."""
        entities = extraction.get("entities", [])
        relations = extraction.get("relations", [])
        with self._driver.session() as session:
            session.execute_write(self._merge_chunk, chunk_id, source)
            for entity in entities:
                session.execute_write(self._merge_entity, chunk_id, entity)
            for relation in relations:
                session.execute_write(self._merge_relation, relation)
        return len(entities), len(relations)

    @staticmethod
    def _merge_chunk(tx, chunk_id: int, source: str) -> None:
        tx.run(
            "MERGE (c:Chunk {chunk_id: $chunk_id}) SET c.source = $source",
            chunk_id=chunk_id,
            source=source,
        )

    @staticmethod
    def _merge_entity(tx, chunk_id: int, entity: dict) -> None:
        tx.run(
            """
            MERGE (e:Entity {name: $name})
            SET e.type = $type
            WITH e
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            name=entity.get("name", "").strip(),
            type=entity.get("type", "Unknown"),
            chunk_id=chunk_id,
        )

    @staticmethod
    def _merge_relation(tx, relation: dict) -> None:
        rel_type = re.sub(r"[^A-Z_]", "_", relation.get("relation", "RELATED").upper()) or "RELATED"
        tx.run(
            f"""
            MERGE (s:Entity {{name: $source}})
            MERGE (t:Entity {{name: $target}})
            MERGE (s)-[r:`{rel_type}`]->(t)
            """,
            source=relation.get("source", "").strip(),
            target=relation.get("target", "").strip(),
        )


def build_graph(chunks: list[Document]) -> GraphStats:
    """Extract a knowledge graph from chunks and write it to Neo4j.

    Args:
        chunks: Chunk documents (each must carry ``chunk_id`` metadata).

    Returns:
        Aggregate :class:`GraphStats` across all chunks.
    """
    stats = GraphStats(chunks=len(chunks))
    with Neo4jGraphWriter() as writer:
        for chunk in chunks:
            chunk_id = int(chunk.metadata.get("chunk_id", 0))
            source = chunk.metadata.get("source", "unknown")
            extraction = extract_graph(chunk.page_content)
            nodes, rels = writer.write_chunk_graph(chunk_id, source, extraction)
            stats.nodes += nodes
            stats.relationships += rels
    logger.info(
        "Inserted %d nodes, %d relationships into Neo4j (from %d chunks)",
        stats.nodes,
        stats.relationships,
        stats.chunks,
    )
    return stats
