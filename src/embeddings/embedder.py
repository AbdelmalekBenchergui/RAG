import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer
from src.helpers import config


_dense_model = None
_sparse_model = None
_sparse_tokenizer = None

MAX_BATCH_SIZE = 16


def _get_dense_model():
    global _dense_model
    if _dense_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _dense_model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    return _dense_model


def _get_sparse_model():
    global _sparse_model, _sparse_tokenizer
    if _sparse_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _sparse_tokenizer = AutoTokenizer.from_pretrained(config.SPARSE_MODEL)
        _sparse_model = AutoModelForMaskedLM.from_pretrained(config.SPARSE_MODEL).to(device)
        _sparse_model.eval()
    return _sparse_model, _sparse_tokenizer


def get_vector_dimension() -> int:
    return _get_dense_model().get_embedding_dimension()


def free_models():
    global _dense_model, _sparse_model, _sparse_tokenizer
    _dense_model = None
    _sparse_model = None
    _sparse_tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_dense_model()
    embeddings = model.encode(texts, batch_size=MAX_BATCH_SIZE, show_progress_bar=False)
    return embeddings.tolist()


def embed_sparse(texts: list[str]) -> list[dict]:
    model, tokenizer = _get_sparse_model()
    device = next(model.parameters()).device

    results = []
    with torch.no_grad():
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i:i + MAX_BATCH_SIZE]
            tokens = tokenizer(
                batch, padding=True, truncation=True,
                return_tensors="pt", max_length=512
            ).to(device)
            output = model(**tokens)
            logits = output.logits
            sparse_weights = torch.log(1 + torch.relu(logits))
            sparse_weights = torch.max(sparse_weights, dim=1).values

            for row in sparse_weights:
                nonzero = torch.nonzero(row).squeeze(-1)
                indices = nonzero.cpu().tolist()
                values = row[nonzero].cpu().tolist()
                if isinstance(indices, int):
                    indices = [indices]
                    values = [values]
                results.append({"indices": indices, "values": values})
        return results
