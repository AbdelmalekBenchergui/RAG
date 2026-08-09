# Mini RAG

A fully **local** Retrieval-Augmented Generation (RAG) system. Ingest your documents (PDF, DOCX, HTML, TXT), ask questions in natural language, and get answers grounded strictly in your own content — no data leaves your machine, no API bills.

It combines two retrieval paradigms (**dense** semantic + **SPLADE** sparse lexical search) fused with **Reciprocal Rank Fusion** in Qdrant, enriches results with an extracted **knowledge graph** in Neo4j, reranks candidates with a **cross-encoder**, and generates answers with a **local LLM** served through Ollama.

---

## Features

- **Hybrid retrieval** — dense (`all-MiniLM-L6-v2`) + sparse (SPLADE) embeddings fused with Reciprocal Rank Fusion in Qdrant.
- **Cross-encoder reranking** — candidates are re-scored with `cross-encoder/ms-marco-MiniLM-L-6-v2` before the top-k is passed to the LLM.
- **Knowledge graph** — entities and relationships extracted by an LLM at ingest, stored in Neo4j, and used to enrich answers with related-graph context.
- **Confidence gating** — if the best reranked score is below a threshold, the system returns "No relevant information found" instead of risking a hallucinated answer.
- **LLM-as-a-judge evaluation** — faithfulness, relevancy, context relevance, answer correctness, completeness, retrieval support, abstention, and latency over a test set.
- **API server** — FastAPI server that keeps models warm and serves queries over HTTP.
- **100% local** — only a one-time model download is needed; afterwards it runs offline (`HF_HUB_OFFLINE=1`).

---

## Architecture

```
                        INGESTION
Documents ──► Loaders ──► Chunker ──► Embedder (dense + SPLADE) ──► Qdrant
(PDF/DOCX/       │        512 tok /        │
 HTML/TXT)       │        50 overlap        │
                 │                          └─► KG Extractor (Ollama LLM) ──► Neo4j

                        QUERY
Query text ──► Embedder (dense + SPLADE) ──► Hybrid Retrieval (RRF) ──► Qdrant
                                                          │
                                                          └────────────► Neo4j (entities + graph expansion)
                                                          │
                                                          ▼
                                              Cross-encoder rerank ──► Ollama LLM ──► Answer + sources
```

### Tech stack

| Component | Role |
| --- | --- |
| **Qdrant** | Vector database: dense + sparse hybrid index, RRF fusion |
| **Neo4j** | Graph database: extracted entities and relationships |
| **Ollama** | Local LLM inference server (`llama3.1`) |
| **sentence-transformers** | Dense embedding model (`all-MiniLM-L6-v2`) + reranker |
| **SPLADE** | Sparse lexical embedding model (`naver/splade-cocondenser-selfdistil`) |
| **FastAPI + uvicorn** | HTTP API server |

---

## Prerequisites

- **Docker** (for Qdrant and Neo4j)
- **Ollama** installed and running with `llama3.1` pulled: `ollama pull llama3.1`
- **Python 3.12+**

---

## Setup

```bash
# 1. Environment configuration
cp .env.example .env           # adjust values as needed

# 2. Python dependencies
pip install -r requirements.txt

# 3. Start the database engines (Qdrant + Neo4j)
python -m src.main up

# 4. Make sure Ollama is running separately
ollama serve                  # in another terminal
```

> **Note:** `.env` sets `HF_HUB_OFFLINE=1`, so embedding models load from the local Hugging Face cache (they must be downloaded once on first run with internet enabled).

---

## Usage (CLI)

```bash
# Start / stop database services (Qdrant + Neo4j via Docker)
python -m src.main up
python -m src.main down

# Ingest a folder of documents (--reset wipes the existing index + graph first)
python -m src.main ingest <folder> --reset

# Ask a question
python -m src.main query "What is the difference between LSM-Trees and B-Trees?" --top-k 5

# Evaluate RAG performance on a test set (LLM-as-a-judge)
python -m src.main eval --queries data/test_queries.json
python -m src.main eval --queries data/test_queries.json --json

# Run the API server (models stay loaded, warm after ~30s)
python -m src.main serve       # http://127.0.0.1:8000
```

---

## API server

Run `python -m src.main serve`, then use the endpoints below.

| Method | Path | Body / Params | Description |
| --- | --- | --- | --- |
| GET | `/health` | — | Status of Qdrant / Neo4j / Ollama |
| GET | `/status` | — | Index counts (Qdrant points, Neo4j entities/relations) + config |
| POST | `/load` | multipart `file` + `save` (bool) | Step 1: load a document. `save=true` also embeds and stores it |
| POST | `/retrieve` | `{"text": "...", "top_k": 5, "rerank": true, "include_entities": true}` | Retrieval pipeline only (embed → search → rerank → entities) |
| POST | `/query` | `{"text": "...", "top_k": 5}` | Full RAG query: retrieve + graph context + LLM answer |

Examples:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/query \
     -H "Content-Type: application/json" \
     -d '{"text":"What is Docker Compose used for?"}'

curl -X POST http://127.0.0.1:8000/load \
     -F "file=@report.pdf" -F "save=true"
```

---

## Configuration

All settings are read from `.env` via `src/helpers/config.py` (see `.env.example`).

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 512 / 50 | chunking |
| `COLLECTION_NAME` | rag_collection | Qdrant collection name |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | dense embedding model |
| `HYBRID_SEARCH_ENABLED` | true | dense + sparse fusion |
| `RERANK_ENABLED` | true | cross-encoder reranking |
| `RERANK_MODEL` | cross-encoder/ms-marco-MiniLM-L-6-v2 | reranker model |
| `RERANK_CANDIDATES` | 20 | candidates fetched before reranking |
| `RETRIEVAL_MIN_SCORE` | 0.1 | confidence threshold (sigmoid scores) |
| `QDRANT_URL` | http://localhost:6333 | vector DB endpoint |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | bolt://localhost:7687 / neo4j / password | graph DB endpoint |
| `OLLAMA_BASE_URL` / `LLM_MODEL` | http://localhost:11434 / llama3.1 | LLM endpoint and model |
| `KG_EXTRACTION_ENABLED` | true | build the knowledge graph at ingest |
| `KG_BATCH_SIZE` / `KG_MAX_CONCURRENCY` | 5 / 4 | KG extraction batching / parallelism |
| `KG_MAX_CHUNKS` | 200 | cap on chunks sent to KG extraction |
| `GRAPH_EXPANSION_ENABLED` / `GRAPH_MAX_HOPS` | true / 1 | graph context expansion at query time |

---

## How it works

**Ingestion**
1. **Load** — each supported file type is parsed into plain text with metadata (`src/loaders`).
2. **Chunk** — long text is split into overlapping chunks of 512 tokens with a 50-token overlap (`src/chunking`).
3. **Embed** — each chunk gets a dense (MiniLM) and a sparse (SPLADE) embedding (`src/embeddings`).
4. **Store** — chunks are upserted into Qdrant under a stable content-derived ID (re-ingesting the same folder is idempotent — no duplicates).
5. **Extract graph** — batches of chunks are sent to the LLM to extract entities and relationships, stored in Neo4j (`src/knowledge_graph`).

**Query**
1. Embed the query (dense + sparse).
2. Hybrid search in Qdrant with Reciprocal Rank Fusion over dense and sparse rankings.
3. Cross-encoder reranking of the candidates.
4. Entity tagging: which graph entities appear in the retrieved chunks (via Neo4j full-text index).
5. Graph expansion: neighbor relationships of those entities are appended to the prompt context.
6. Ollama generates the final answer grounded in the context; sources are returned with it.

---

## Evaluation

The system evaluates itself with an **LLM-as-a-judge** approach. Each test query is
run through the full RAG pipeline and scored by a judge model (`JUDGE_MODEL`,
defaults to `LLM_MODEL`; `EVAL_JUDGE_PASSES` controls judge self-consistency, default 2).

```
python -m src.main eval --queries data/test_queries.json
python -m src.main eval --queries data/test_queries.json --json   # full report as JSON
```

Test-set schema (backward compatible — a plain `"question"` also works):

```json
{"question": "...", "answer": "reference answer (optional)", "category": "factual"}
```

Categories: `factual`, `relational` (exercises the knowledge graph), `multi-hop`, `negative`
(out-of-corpus, used to test confidence gating).

| Metric | Type | What it measures |
| --- | --- | --- |
| Faithfulness (0–5) | judge | answer grounded in the retrieved context |
| Relevancy (0–5) | judge | answer directly addresses the question |
| Context relevance (0–5) | judge | retrieved context is relevant/sufficient for the question |
| Retrieval support (0/1) | judge | context contains the information to answer |
| Answer correctness (0–5) | judge + ground truth | semantic match vs reference answer |
| Completeness (0–1) | judge + ground truth | fraction of reference facts covered |
| Abstention rate | computed | % of queries where confidence gating fired |
| Abstained-correctly | computed | negative queries that were correctly rejected |
| Latency | computed | total, retrieve, and generate times (per-stage from the query trace) |
| Retrieval score | computed | mean / max reranked score across queries |

The report also prints a **per-category breakdown** and the **worst-performing queries**.

> **Note:** the test set (`data/`) is gitignored, so on a fresh clone point `eval` at your own file:
> `python -m src.main eval --queries path/to/queries.json`.

---

## Project structure

```
src/
├── loaders/            # PDF, DOCX, HTML, TXT parsers
├── chunking/           # text splitting
├── embeddings/         # dense (MiniLM), sparse (SPLADE), reranker
├── vector_store/       # Qdrant collection lifecycle + hybrid search
├── knowledge_graph/    # entity/relationship extraction + Neo4j storage
├── controllers/        # end-to-end query orchestration
├── evaluation/         # LLM-as-a-judge scoring
├── pipeline/           # ingestion orchestration
├── helpers/            # configuration + health checks
├── server.py           # FastAPI application
└── main.py             # CLI entry point
```

---
