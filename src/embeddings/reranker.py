import torch
from sentence_transformers import CrossEncoder
from src.helpers import config


_reranker = None
_load_failed = False


def _get_reranker():
    global _reranker, _load_failed
    if _reranker is None and not _load_failed:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _reranker = CrossEncoder(config.RERANK_MODEL, device=device)
        except Exception as e:
            _load_failed = True
            print(f"[WARN] Reranker unavailable ({config.RERANK_MODEL}): {e}. Falling back to vector scores.")
    return _reranker


def rerank(query: str, results: list[dict], top_k: int) -> list[dict]:
    if not results:
        return []

    model = _get_reranker()
    if model is None:
        return results[:top_k]

    pairs = [(query, r["text"]) for r in results]
    scores = model.predict(pairs, show_progress_bar=False, convert_to_tensor=True)
    scores = torch.sigmoid(scores).tolist()

    scored = list(zip(results, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    ranked = []
    for r, score in scored[:top_k]:
        r = dict(r)
        r["score"] = float(score)
        ranked.append(r)
    return ranked
