import hashlib
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
from src.helpers import config
from src.embeddings import embedder


_client = None
_collection_ready = False


def _get_client():
    global _client
    if _client is None:
        _client = QdrantClient(url=config.QDRANT_URL)
    return _client


def _vector_dim() -> int:
    return embedder.get_vector_dimension()


def init_collection(reset: bool = False):
    global _collection_ready
    client = _get_client()
    collection_name = config.COLLECTION_NAME

    try:
        client.get_collection(collection_name)
        if reset:
            client.delete_collection(collection_name)
            _create_collection(client, collection_name)
            print(f"Reset collection '{collection_name}'")
        else:
            print(f"Using existing collection '{collection_name}'")
    except UnexpectedResponse:
        _create_collection(client, collection_name)
        print(f"Created collection '{collection_name}'")

    _collection_ready = True


def _create_collection(client, name: str):
    vectors_config = {
        "dense": qmodels.VectorParams(
            size=_vector_dim(),
            distance=qmodels.Distance.COSINE,
        ),
    }
    sparse_vectors_config = None
    if config.HYBRID_SEARCH_ENABLED:
        sparse_vectors_config = {
            "sparse": qmodels.SparseVectorParams(),
        }

    client.create_collection(
        collection_name=name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )


def _chunk_id(text: str, meta: dict) -> str:
    filename = meta.get("filename") or "unknown"
    chunk_index = meta.get("chunk_index")
    if chunk_index is not None:
        key = f"{filename}\x00{chunk_index}"
    else:
        key = text
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return str(uuid.UUID(bytes=digest[:16]))


def add_chunks(texts: list[str], embeddings: list[list[float]], metadatas: list[dict],
               sparse_embeddings: list[dict] | None = None):
    if not _collection_ready:
        init_collection()
    client = _get_client()

    points = []
    for i, (text, emb, meta) in enumerate(zip(texts, embeddings, metadatas)):
        vector = {"dense": emb}
        if sparse_embeddings is not None:
            se = sparse_embeddings[i]
            vector["sparse"] = qmodels.SparseVector(
                indices=se["indices"],
                values=se["values"],
            )
        points.append(
            qmodels.PointStruct(
                id=_chunk_id(text, meta),
                vector=vector,
                payload={"text": text, **meta},
            )
        )

    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=config.COLLECTION_NAME,
            points=points[i:i + batch_size],
        )


def search(query_embedding: list[float], top_k: int = 5,
           query_sparse_embedding: list[float] | None = None,
           candidates: int | None = None) -> list[dict]:
    if not _collection_ready:
        init_collection()
    client = _get_client()

    limit = candidates if candidates is not None else top_k
    limited = config.HYBRID_SEARCH_ENABLED and query_sparse_embedding is not None

    if limited:
        response = client.query_points(
            collection_name=config.COLLECTION_NAME,
            prefetch=[
                qmodels.Prefetch(
                    query=query_embedding,
                    limit=limit * 10,
                    using="dense",
                ),
                qmodels.Prefetch(
                    query=qmodels.SparseVector(
                        indices=query_sparse_embedding["indices"],
                        values=query_sparse_embedding["values"],
                    ),
                    limit=limit * 10,
                    using="sparse",
                ),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=limit,
        )
    else:
        response = client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=query_embedding,
            limit=limit,
            using="dense",
        )
    results = response.points

    output = []
    for hit in results:
        payload = hit.payload or {}
        text = payload.pop("text", "")
        output.append(
            {
                "id": hit.id,
                "text": text,
                "metadata": payload,
                "score": hit.score,
            }
        )
    return output
