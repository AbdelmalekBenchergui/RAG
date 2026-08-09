import time

import httpx
from src.embeddings import embedder
from src.helpers import config

NO_ANSWER = "No relevant information found in the indexed documents."


def _trace(trace: list, t0: float, **fields) -> None:
    if trace is None:
        return
    step = dict(fields)
    step["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    trace.append(step)


def _generate(prompt: str, context: str) -> str:
    full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer based on the context above:"
    payload = {
        "model": config.LLM_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    resp = httpx.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _enrich_with_entities(results: list[dict]) -> None:
    from src.knowledge_graph.neo4j_store import query_entities_in_text

    context_text = "\n".join(r["text"] for r in results)
    context_entities = query_entities_in_text(context_text)
    if not context_entities:
        return
    for r in results:
        lowered = r["text"].lower()
        r["related_entities"] = [e for e in context_entities if e["name"].lower() in lowered]


def _graph_context(context_entities: list[dict]) -> str:
    from src.knowledge_graph.neo4j_store import query_related

    seen = set()
    lines = []
    for entity in context_entities:
        name = entity["name"]
        if name in seen:
            continue
        seen.add(name)
        if len(seen) > config.GRAPH_EXPANSION_MAX_ENTITIES:
            break
        related = query_related(name, max_hops=config.GRAPH_MAX_HOPS, limit=10)
        for rel in related:
            relation = rel.get("relation") or "RELATED"
            lines.append(
                f"{name} --[{relation}]--> {rel['name']} ({rel.get('type', '?')})"
            )
        if len(lines) >= 30:
            break
    if not lines:
        return ""
    return "Related graph context:\n" + "\n".join(lines)


def retrieve(text: str, top_k: int = 5, rerank: bool | None = None,
             include_entities: bool | None = None, trace: list | None = None) -> list[dict]:
    t0 = time.perf_counter()
    if rerank is None:
        rerank = config.RERANK_ENABLED
    if include_entities is None:
        include_entities = config.KG_EXTRACTION_ENABLED

    embedding = embedder.embed_texts([text])[0]

    sparse_embedding = None
    if config.HYBRID_SEARCH_ENABLED:
        sparse_embedding = embedder.embed_sparse([text])[0]

    _trace(trace, t0, step="embed_query",
           vector_dim=len(embedding), sparse=sparse_embedding is not None)

    from src.vector_store.qdrant_store import search as vector_search

    candidates = config.RERANK_CANDIDATES if rerank else top_k
    results = vector_search(
        embedding, top_k=candidates, query_sparse_embedding=sparse_embedding, candidates=candidates
    )

    _trace(trace, t0, step="search", candidates=len(results))

    if rerank:
        from src.embeddings.reranker import rerank as rerank_results
        results = rerank_results(text, results, top_k=top_k)
        _trace(trace, t0, step="rerank", top_k=len(results))

    if include_entities:
        _enrich_with_entities(results)
        with_entities = sum(1 for r in results if r.get("related_entities"))
        _trace(trace, t0, step="entities", results_with_entities=with_entities)

    return results


def query(text: str, top_k: int = 5, trace: list | None = None) -> dict:
    t0 = time.perf_counter()
    if not text.strip():
        return {"answer": "Please provide a non-empty question.", "results": [], "confident": False}

    results = retrieve(text, top_k=top_k, trace=trace)

    if not results:
        return {"answer": NO_ANSWER, "results": [], "confident": False}

    if results[0]["score"] < config.RETRIEVAL_MIN_SCORE:
        return {
            "answer": NO_ANSWER,
            "results": results,
            "confident": False,
        }

    context_entities = []
    seen = set()
    for r in results:
        for e in r.get("related_entities", []):
            if e["name"] not in seen:
                seen.add(e["name"])
                context_entities.append(e)

    context = "\n\n".join(
        f"Source: {r['metadata'].get('filename', 'unknown')}\n{r['text']}"
        for r in results
    )

    graph_lines = 0
    if config.GRAPH_EXPANSION_ENABLED and context_entities:
        graph_ctx = _graph_context(context_entities)
        if graph_ctx:
            context = f"{context}\n\n{graph_ctx}"
            graph_lines = len(graph_ctx.splitlines())

    _trace(trace, t0, step="graph_context", lines=graph_lines, entities=len(context_entities))

    gen_t0 = time.perf_counter()
    answer = _generate(text, context)

    if trace is not None:
        trace.append({
            "step": "generate",
            "model": config.LLM_MODEL,
            "elapsed_ms": round((time.perf_counter() - gen_t0) * 1000, 1),
        })

    return {"answer": answer, "results": results, "confident": True}
