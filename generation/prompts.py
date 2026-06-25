"""Prompt templates for the RAG answer-generation step.

The system prompt instructs the model to ground every claim in the supplied
context and to cite sources using a ``[source: <name>#<chunk_id>]`` convention.
A short few-shot example demonstrates the desired citation format.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are GraphRAG Assistant, a precise question-answering system over a \
document corpus. You are given two kinds of context:

1. VECTOR CONTEXT — passages retrieved by semantic similarity.
2. GRAPH CONTEXT — facts retrieved from a knowledge graph (entities and their \
   relationships), useful for multi-hop reasoning.

Rules:
- Answer ONLY from the provided context. If the answer is not present, say you \
  don't have enough information.
- Cite every factual claim with its source in the form [source: <file>#<chunk_id>].
- Prefer combining vector and graph context when a question spans multiple entities.
- Be concise and lead with the direct answer.
"""

# A single few-shot exchange to anchor the citation style.
FEW_SHOT_EXAMPLE = """Example
-------
VECTOR CONTEXT:
[source: handbook.md#3] Acme Corp was founded in 1998 by Dana Lee.

GRAPH CONTEXT:
(Dana Lee)-[:FOUNDED]->(Acme Corp); (Acme Corp)-[:HEADQUARTERED_IN]->(Berlin)

QUESTION: Who founded Acme Corp and where is it based?

ANSWER: Acme Corp was founded by Dana Lee in 1998 [source: handbook.md#3], and \
it is headquartered in Berlin [source: graph].
"""

ANSWER_PROMPT_TEMPLATE = """{few_shot}

Now answer the user's question using the context below.

VECTOR CONTEXT:
{vector_context}

GRAPH CONTEXT:
{graph_context}

QUESTION: {question}

ANSWER:"""


def build_answer_prompt(question: str, vector_context: str, graph_context: str) -> str:
    """Render the final user-turn prompt for answer generation."""
    return ANSWER_PROMPT_TEMPLATE.format(
        few_shot=FEW_SHOT_EXAMPLE,
        vector_context=vector_context or "(none)",
        graph_context=graph_context or "(none)",
        question=question,
    )
