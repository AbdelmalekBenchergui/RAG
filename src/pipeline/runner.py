import os
import time
from pathlib import Path
from src.helpers import config
from src.loaders import pdf, docx, html, txt
from src.chunking import chunker
from src.embeddings import embedder

SUPPORTED_EXTENSIONS = {
    ".pdf": pdf.parse_pdf_bytes,
    ".docx": docx.parse_docx_bytes,
    ".html": html.parse_html_bytes,
    ".htm": html.parse_html_bytes,
    ".txt": txt.parse_txt_bytes,
}


def scan_files(directory: str) -> list[Path]:
    files = []
    for root, _, filenames in os.walk(directory):
        for f in filenames:
            path = Path(root) / f
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)
    return files


def load_document(filepath: Path):
    ext = filepath.suffix.lower()
    loader = SUPPORTED_EXTENSIONS[ext]
    with open(filepath, "rb") as f:
        content = f.read()
    text, meta = loader(content, str(filepath))
    return text, meta


def run(directory: str, reset: bool = False):
    from src.vector_store.qdrant_store import init_collection as qdrant_init, add_chunks as qdrant_add

    qdrant_init(reset=reset)

    if config.KG_EXTRACTION_ENABLED:
        from src.knowledge_graph.neo4j_store import init_store as kg_init, store_extractions
        kg_init(reset=reset)

    files = scan_files(directory)
    print(f"Found {len(files)} supported files in {directory}")

    all_chunks = []
    start_time = time.time()

    for i, filepath in enumerate(files, 1):
        try:
            text, meta = load_document(filepath)
            chunks = chunker.split_text(text)
            for c in chunks:
                c["metadata"].update(meta)
            all_chunks.extend(chunks)
            elapsed = time.time() - start_time
            print(f"  [{i}/{len(files)}] {filepath.name} -> {len(chunks)} chunks ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  [{i}/{len(files)}] {filepath.name} -> FAILED: {e}")

    if not all_chunks:
        print("No chunks to index.")
        return

    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    # --- Branch A: Vector embeddings -> Vector DB ---
    print(f"Embedding {len(texts)} chunks...")
    embed_start = time.time()
    embeddings = embedder.embed_texts(texts)
    print(f"Embedding done ({time.time() - embed_start:.1f}s)")

    sparse_embeddings = None
    if config.HYBRID_SEARCH_ENABLED:
        print(f"Generating sparse embeddings ({len(texts)} chunks)...")
        sparse_start = time.time()
        sparse_embeddings = embedder.embed_sparse(texts)
        print(f"Sparse embedding done ({time.time() - sparse_start:.1f}s)")

    print("Storing in Qdrant...")
    qdrant_add(texts, embeddings, metadatas, sparse_embeddings=sparse_embeddings)
    print("Vector indexing done")
    embedder.free_models()

    # --- Branch B: Knowledge Graph extraction -> Neo4j ---
    if config.KG_EXTRACTION_ENABLED:
        print("Extracting knowledge graph...")
        kg_start = time.time()
        from concurrent.futures import ThreadPoolExecutor
        from src.knowledge_graph.extractor import extract_batch

        batch_size = config.KG_BATCH_SIZE
        kg_texts = texts[:config.KG_MAX_CHUNKS]
        if len(kg_texts) < len(texts):
            print(f"  Limiting KG extraction to first {len(kg_texts)} chunks "
                  f"(KG_MAX_CHUNKS={config.KG_MAX_CHUNKS})")
        batches = [
            kg_texts[i:i + batch_size]
            for i in range(0, len(kg_texts), batch_size)
        ]

        all_extractions = [None] * len(batches)
        done = 0
        with ThreadPoolExecutor(max_workers=config.KG_MAX_CONCURRENCY) as pool:
            futures = {
                pool.submit(extract_batch, batch_texts): (idx, batch_texts)
                for idx, batch_texts in enumerate(batches)
            }
            for future in futures:
                idx, _ = futures[future]
                try:
                    all_extractions[idx] = future.result()
                except Exception as e:
                    print(f"  KG batch {idx} failed: {e}")
                    all_extractions[idx] = []
                done += 1
                print(f"  KG batch [{done}/{len(batches)}]")

        all_extractions = [ex for batch in all_extractions for ex in batch]
        print(f"KG extraction done ({time.time() - kg_start:.1f}s)")

        print("Storing in Neo4j...")
        store_extractions(all_extractions)
        print(f"Graph indexing done")

    total_time = time.time() - start_time
    print(f"Done! Indexed {len(texts)} chunks from {len(files)} documents in {total_time:.1f}s")
