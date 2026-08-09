from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.helpers import config


class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)


class RetrieveRequest(BaseModel):
    text: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    rerank: bool | None = None
    include_entities: bool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Warming up models (one-time, ~25s)...")
    from src.embeddings import embedder
    from src.embeddings.reranker import _get_reranker

    embedder.embed_texts(["warmup"])
    if config.HYBRID_SEARCH_ENABLED:
        embedder.embed_sparse(["warmup"])
    if config.RERANK_ENABLED:
        _get_reranker()
    print("Models ready.")
    yield
    from src.knowledge_graph.neo4j_store import close
    close()


app = FastAPI(title="Mini RAG", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    from src.helpers import health as h

    checks = {
        "qdrant": h.check_qdrant() or "ok",
        "neo4j": h.check_neo4j() or "ok",
        "ollama": h.check_ollama() or "ok",
    }
    healthy = all(v == "ok" for v in checks.values())
    return {"status": "ok" if healthy else "degraded", "checks": checks}


@app.get("/status")
def status():
    from src.helpers import config

    payload = {"config": {}}
    for key in (
        "COLLECTION_NAME", "CHUNK_SIZE", "CHUNK_OVERLAP",
        "EMBEDDING_MODEL", "SPARSE_MODEL", "RERANK_MODEL",
        "HYBRID_SEARCH_ENABLED", "RERANK_ENABLED", "KG_EXTRACTION_ENABLED",
        "GRAPH_EXPANSION_ENABLED", "RETRIEVAL_MIN_SCORE", "LLM_MODEL",
    ):
        payload["config"][key] = getattr(config, key)

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=config.QDRANT_URL)
        info = client.get_collection(config.COLLECTION_NAME)
        vc = info.config.params.vectors
        vector_names = list(vc.keys()) if isinstance(vc, dict) else ["default"]
        if getattr(info.config.params, "sparse_vectors", None):
            vector_names.append("sparse")
        payload["qdrant"] = {"points": info.points_count, "vectors": vector_names}
    except Exception as e:
        payload["qdrant"] = {"error": str(e)}

    try:
        from src.knowledge_graph.neo4j_store import _get_driver
        with _get_driver().session() as session:
            payload["neo4j"] = {
                "entities": session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"],
                "relationships": session.run("MATCH ()-[r:RELATED]->() RETURN count(r) AS n").single()["n"],
            }
    except Exception as e:
        payload["neo4j"] = {"error": str(e)}

    return payload


@app.post("/load")
async def load_docs(file: UploadFile = File(...), save: bool = Form(False)):
    """Step 1: load a document (file upload).

    save=true  -> also embed + store the chunks in Qdrant (steps 2-3).
    """
    from src.pipeline.runner import SUPPORTED_EXTENSIONS
    from src.chunking import chunker
    from src.embeddings import embedder
    from src.vector_store.qdrant_store import add_chunks

    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported extension '{ext}'")

    documents = []
    try:
        content = await file.read()
        loader = SUPPORTED_EXTENSIONS[ext]
        text, meta = loader(content, filename)
        chunks = chunker.split_text(text)
        for c in chunks:
            c["metadata"].update(meta)
        documents.append({
            "filename": filename,
            "type": meta.get("type"),
            "num_chunks": len(chunks),
            "chunks": [
                {"text": c["text"], "chunk_index": c["metadata"]["chunk_index"], "metadata": c["metadata"]}
                for c in chunks
            ],
        })
    except Exception as e:
        documents.append({"filename": filename, "error": str(e)})

    steps = [{"step": "load", "documents": documents}]

    if save:
        texts, metadatas = [], []
        for doc in documents:
            for c in doc.get("chunks", []):
                texts.append(c["text"])
                metadatas.append(c["metadata"])
        if texts:
            dense = embedder.embed_texts(texts)
            sparse = None
            if config.HYBRID_SEARCH_ENABLED:
                sparse = embedder.embed_sparse(texts)
            steps.append({
                "step": "embed",
                "num_texts": len(texts),
                "vector_dim": len(dense[0]),
                "sparse": sparse is not None,
            })
            add_chunks(texts, dense, metadatas, sparse_embeddings=sparse)
            steps.append({"step": "store", "stored": len(texts)})
        else:
            steps.append({"step": "embed", "status": "skipped", "reason": "no chunks"})
            steps.append({"step": "store", "status": "skipped", "reason": "no chunks"})

    return {"status": "ok", "save": save, "steps": steps}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    """Retrieval: embed the query, search the vector DB, rerank, add graph entities."""
    from src.controllers.hybrid_query import retrieve as retrieve_controller

    try:
        steps = []
        results = retrieve_controller(
            req.text,
            top_k=req.top_k,
            rerank=req.rerank,
            include_entities=req.include_entities,
            trace=steps,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")
    return {"steps": steps, "results": results}


@app.post("/query")
def query(req: QueryRequest):
    """Full RAG: retrieve (steps) + graph context + LLM answer."""
    from src.controllers.hybrid_query import query as query_controller

    try:
        steps = []
        result = query_controller(req.text, top_k=req.top_k, trace=steps)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
    return {
        "steps": steps,
        "answer": result.get("answer", ""),
        "confident": result.get("confident", True),
        "results": result.get("results", []),
    }
