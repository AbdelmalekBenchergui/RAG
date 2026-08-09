import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

if os.getenv("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes"):
    os.environ["HF_HUB_OFFLINE"] = "1"

SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "naver/splade-cocondenser-selfdistil")
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"

RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))

RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.1"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "") or LLM_MODEL
EVAL_JUDGE_PASSES = int(os.getenv("EVAL_JUDGE_PASSES", "2"))

KG_EXTRACTION_ENABLED = os.getenv("KG_EXTRACTION_ENABLED", "true").lower() == "true"
KG_BATCH_SIZE = int(os.getenv("KG_BATCH_SIZE", "5"))
KG_MAX_CONCURRENCY = int(os.getenv("KG_MAX_CONCURRENCY", "4"))
KG_MAX_CHUNKS = int(os.getenv("KG_MAX_CHUNKS", "200"))

GRAPH_EXPANSION_ENABLED = os.getenv("GRAPH_EXPANSION_ENABLED", "true").lower() == "true"
GRAPH_MAX_HOPS = int(os.getenv("GRAPH_MAX_HOPS", "1"))
GRAPH_EXPANSION_MAX_ENTITIES = int(os.getenv("GRAPH_EXPANSION_MAX_ENTITIES", "5"))
