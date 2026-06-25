# GraphRAG Docs Assistant

![python](https://img.shields.io/badge/python-3.11-blue)
![framework](https://img.shields.io/badge/framework-LangChain-1c3c3c)
![vector](https://img.shields.io/badge/vector-ChromaDB-orange)
![graph](https://img.shields.io/badge/graph-Neo4j-008cc1)
![api](https://img.shields.io/badge/api-FastAPI-009688)
![llm](https://img.shields.io/badge/LLM-claude--opus--4--8%20%7C%20Ollama-8a2be2)
![license](https://img.shields.io/badge/license-MIT-green)

A **hybrid Retrieval-Augmented Generation** assistant over a document corpus. It
combines **dense vector search** (ChromaDB) with a **knowledge graph** (Neo4j)
so it can answer both "what does the doc say about X" questions *and* multi-hop
questions that require connecting entities across the corpus. Orchestration is
done with **LangChain**, generation runs on **Anthropic `claude-opus-4-8`** (or a
local **Ollama** model), conversation/metadata is persisted in **PostgreSQL**,
and everything is served through a **FastAPI** backend with a minimal HTML/JS
chat frontend. The whole stack is **dockerized**.

---

## Architecture

```
                           ┌──────────────────────────────────────────┐
                           │              FastAPI (api/)                │
   Browser  ──HTTP──▶      │  GET /health   POST /chat   POST /ingest   │
  (frontend/index.html)    │  POST /feedback         GET /  (frontend)  │
                           └───────┬───────────────────────┬───────────┘
                                   │                        │
                  ingest pipeline  │                        │  chat pipeline
                                   ▼                        ▼
        ┌──────────────────────────────────┐     ┌────────────────────────────┐
        │          ingestion/              │     │         retrieval/         │
        │  loader  → chunks                │     │  vector_retriever (Chroma) │
        │  embeddings → ChromaDB           │     │  graph_retriever  (Cypher) │
        │  graph_builder → Neo4j (LLM)     │     │  hybrid → merged context   │
        └───────────┬──────────────┬───────┘     └──────────────┬─────────────┘
                    │              │                            │
                    ▼              ▼                            ▼
             ┌───────────┐   ┌──────────┐              ┌─────────────────┐
             │ ChromaDB  │   │  Neo4j   │              │  generation/    │
             │ (vectors) │   │ (graph)  │              │  prompts + llm  │
             └───────────┘   └──────────┘              │  claude-opus-4-8│
                                                       │     or Ollama   │
                    ┌────────────────────────┐         └─────────────────┘
                    │   PostgreSQL (db/)      │
                    │ chat_sessions, messages │◀── conversation + doc metadata
                    │ documents, feedback     │
                    └────────────────────────┘
```

**Hybrid retrieval flow** (`retrieval/hybrid.py`):

1. Vector search retrieves the top-_k_ most similar chunks from ChromaDB.
2. Candidate entity tokens (from the question + top chunks) are matched against
   the Neo4j entity index to find **seed entities**.
3. The graph is expanded around those seeds to pull in related facts — this is
   what enables **multi-hop** reasoning.
4. Both context blocks are handed to the LLM, which answers with **citations**.

---

## Project layout

```
graphrag-docs-assistant/
├── config.py                 # env-driven settings (single source of truth)
├── ingestion/
│   ├── loader.py             # load PDFs/markdown/txt, chunk
│   ├── embeddings.py         # sentence-transformers -> Chroma
│   └── graph_builder.py      # LLM entity/relation extraction -> Neo4j
├── retrieval/
│   ├── vector_retriever.py   # Chroma similarity search
│   ├── graph_retriever.py    # Cypher queries against Neo4j
│   └── hybrid.py             # combine vector + graph (multi-hop)
├── generation/
│   ├── prompts.py            # system prompt, citation format, few-shot
│   └── llm.py                # LLM wrapper (Anthropic claude-opus-4-8 / Ollama)
├── db/
│   ├── schema.sql            # chat_sessions, messages, documents, feedback
│   └── store.py              # psycopg2 helpers
├── api/main.py               # FastAPI app, serves frontend
├── frontend/index.html       # minimal chat UI (vanilla JS)
├── docs/sample.md            # sample corpus
└── notebooks/01_pipeline_demo.ipynb
```

---

## Quick start (Docker)

```bash
git clone https://github.com/KoluzanovP/graphrag-docs-assistant.git
cd graphrag-docs-assistant

cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY, NEO4J_PASSWORD, POSTGRES_PASSWORD

docker compose up --build
```

Services started:

| Service    | URL                         | Notes                          |
|------------|-----------------------------|--------------------------------|
| app        | http://localhost:8000       | FastAPI + chat frontend        |
| neo4j      | http://localhost:7474       | Neo4j Browser (bolt :7687)     |
| postgres   | localhost:5432              | schema auto-applied on init    |
| chroma     | http://localhost:8001       | standalone Chroma server       |

Then open **http://localhost:8000**, click **Ingest docs**, and start asking
questions.

---

## Quick start (local, without Docker)

```bash
python -m venv .venv
source .venv/Scripts/activate        # Git Bash on Windows
# or: .venv\Scripts\Activate.ps1     # PowerShell

pip install -r requirements.txt
cp .env.example .env                  # then edit values

# You still need Neo4j + PostgreSQL reachable at the configured URIs.
uvicorn api.main:app --reload --port 8000
```

---

## Usage

### Ingest

```bash
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{}'
# -> {"documents":1,"chunks":14,"nodes":23,"relationships":31}
```

### Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Who leads the PathWeaver team and who do they report to?"}'
```

```json
{
  "session_id": "b2f1...",
  "message_id": 2,
  "answer": "The PathWeaver team is led by Sofia Reyes [source: sample.md#6], who reports to Idris Kane, the CTO [source: graph].",
  "citations": ["[source: sample.md#6]", "[source: graph]"],
  "graph_facts": ["(Sofia Reyes)-[:LEADS]->(PathWeaver)", "(Sofia Reyes)-[:REPORTS_TO]->(Idris Kane)"]
}
```

### Feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"message_id":2,"rating":1,"comment":"accurate"}'
```

---

## Environment variables

| Variable            | Default                                          | Description                                          |
|---------------------|--------------------------------------------------|------------------------------------------------------|
| `LLM_PROVIDER`      | `anthropic`                                      | `anthropic` or `ollama`                              |
| `ANTHROPIC_API_KEY` | —                                                | Anthropic API key (required when provider=anthropic) |
| `ANTHROPIC_MODEL`   | `claude-opus-4-8`                                | Anthropic model id (default is `claude-opus-4-8`)    |
| `OLLAMA_BASE_URL`   | `http://localhost:11434`                         | Ollama server URL                                    |
| `OLLAMA_MODEL`      | `llama3.1`                                        | Ollama model id                                      |
| `EMBEDDING_MODEL`   | `sentence-transformers/all-MiniLM-L6-v2`         | sentence-transformers model                          |
| `CHROMA_DIR`        | `./.chroma`                                       | Chroma persistence directory                         |
| `CHROMA_COLLECTION` | `graphrag_docs`                                  | Chroma collection name                               |
| `NEO4J_URI`         | `bolt://localhost:7687`                          | Neo4j bolt URI                                       |
| `NEO4J_USER`        | `neo4j`                                          | Neo4j user                                           |
| `NEO4J_PASSWORD`    | —                                                | Neo4j password                                       |
| `POSTGRES_HOST`     | `localhost`                                      | PostgreSQL host                                      |
| `POSTGRES_PORT`     | `5432`                                           | PostgreSQL port                                      |
| `POSTGRES_DB`       | `graphrag`                                        | Database name                                        |
| `POSTGRES_USER`     | `graphrag`                                         | Database user                                        |
| `POSTGRES_PASSWORD` | —                                                | Database password                                    |
| `DOCS_DIR`          | `./docs`                                          | Directory scanned by the loader                      |
| `CHUNK_SIZE`        | `800`                                            | Chunk size (characters)                              |
| `CHUNK_OVERLAP`     | `120`                                            | Chunk overlap (characters)                           |

---

## How the LLM is selected

`generation/llm.py` reads `LLM_PROVIDER`:

- **`anthropic`** → `ChatAnthropic(model="claude-opus-4-8", ...)` from
  `langchain-anthropic`. The default model id is **`claude-opus-4-8`**.
- **`ollama`** → `ChatOllama(model=..., base_url=...)` from
  `langchain-community`, for fully local inference.

The same wrapper is used both for **answer generation** and for **LLM-based
entity/relation extraction** during ingestion.

---

## Notebook

`notebooks/01_pipeline_demo.ipynb` walks through the entire pipeline end to end
— loading & chunking, building embeddings into Chroma, extracting entities into
Neo4j, running a hybrid query, and generating a cited answer — with executed
outputs preserved.

---

## License

MIT.
