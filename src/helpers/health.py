import httpx
from src.helpers import config


def check_qdrant() -> str | None:
    try:
        resp = httpx.get(f"{config.QDRANT_URL}/collections", timeout=5)
        if resp.status_code == 200:
            return None
        return f"Qdrant returned HTTP {resp.status_code}"
    except Exception as e:
        return f"{config.QDRANT_URL} unreachable ({e.__class__.__name__})"


def check_neo4j() -> str | None:
    try:
        from src.knowledge_graph.neo4j_store import _get_driver
        driver = _get_driver()
        driver.verify_connectivity()
        return None
    except Exception as e:
        return f"{config.NEO4J_URI} unreachable ({e.__class__.__name__})"


def check_ollama() -> str | None:
    try:
        resp = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = {m["name"].split(":")[0] for m in resp.json().get("models", [])}
            if config.LLM_MODEL.split(":")[0] not in models:
                return f"model '{config.LLM_MODEL}' not pulled (ollama pull {config.LLM_MODEL})"
            return None
        return f"Ollama returned HTTP {resp.status_code}"
    except Exception as e:
        return f"{config.OLLAMA_BASE_URL} unreachable ({e.__class__.__name__})"


def ensure_services(require_ollama: bool = True, require_kg: bool = False) -> None:
    failures = []
    q = check_qdrant()
    if q:
        failures.append(f"Qdrant: {q}")
    if require_kg:
        n = check_neo4j()
        if n:
            failures.append(f"Neo4j: {n}")
    if require_ollama:
        o = check_ollama()
        if o:
            failures.append(f"Ollama: {o}")
    if failures:
        raise SystemExit(
            "\n".join([f"[ERROR] {f}" for f in failures])
            + "\n\nStart services with: python -m src.main up"
            + "\n(and keep Ollama running separately: ollama serve)\n"
        )
