import argparse
from src.helpers import config
from src.helpers.health import ensure_services
from src.pipeline import ingestion
from src.controllers.hybrid_query import query as query_controller


def main():
    parser = argparse.ArgumentParser(description="Mini RAG - Local RAG ingestion system")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest documents from a directory")
    ingest.add_argument("directory", help="Path to directory containing documents")
    ingest.add_argument("--reset", action="store_true", help="Reset the vector index and graph before ingesting")

    query = subparsers.add_parser("query", help="Query the indexed documents")
    query.add_argument("text", help="Search query")
    query.add_argument("--top-k", type=int, default=5, help="Number of results to return")

    eval_parser = subparsers.add_parser("eval", help="Evaluate RAG performance on a test set")
    eval_parser.add_argument("--queries", default="data/test_queries.json", help="Path to JSON test queries file")
    eval_parser.add_argument("--json", action="store_true", help="Output results as JSON")

    up = subparsers.add_parser("up", help="Start Docker services (Qdrant, Neo4j)")
    down = subparsers.add_parser("down", help="Stop Docker services")
    serve = subparsers.add_parser("serve", help="Run the FastAPI server (models stay loaded)")

    args = parser.parse_args()

    if args.command == "ingest":
        ensure_services(require_ollama=config.KG_EXTRACTION_ENABLED, require_kg=config.KG_EXTRACTION_ENABLED)
        ingestion.run(args.directory, reset=args.reset)
    elif args.command == "query":
        ensure_services(require_kg=config.KG_EXTRACTION_ENABLED)
        result = query_controller(args.text, top_k=args.top_k)
        answer = result.get("answer", "")
        print(f"Answer:\n{answer}\n")
        if not result.get("confident", True):
            print("[NOTE] Low retrieval confidence - the answer above did not use the LLM.\n")
        print("--- Sources ---")
        for r in result.get("results", []):
            score = r["score"]
            meta = r.get("metadata", {})
            print(f"[{score:.4f}] {meta.get('filename', 'unknown')}")
            print(f"   {r['text'][:300]}")
            if r.get("related_entities"):
                names = [e["name"] for e in r["related_entities"]]
                print(f"   Entities: {names}")
            print()
    elif args.command == "eval":
        from src.evaluation.evaluator import evaluate
        ensure_services(require_kg=config.KG_EXTRACTION_ENABLED)
        result = evaluate(args.queries)
        s = result["summary"]
        print(f"\n{'='*60}")
        print(f"Evaluation Summary ({s['num_queries']} queries, "
              f"{s['num_with_ground_truth']} with ground truth)")
        print(f"{'='*60}")
        print(f"  Avg Faithfulness:        {s['avg_faithfulness']:.2f} / 5")
        print(f"  Avg Relevancy:           {s['avg_relevancy']:.2f} / 5")
        print(f"  Avg Context Relevance:   {s['avg_context_relevance']:.2f} / 5")
        print(f"  Retrieval Support Rate:  {s['retrieval_support_rate']:.2f}")
        print(f"  Avg Answer Correctness:  {s['avg_answer_correctness']:.2f} / 5")
        print(f"  Avg Completeness:        {s['avg_completeness']:.2f}")
        print(f"  Abstention Rate:         {s['abstention_rate']:.2f}")
        if s["abstained_correctly_rate"] is not None:
            print(f"  Negative queries correct: {s['abstained_correctly_rate']:.2f}")
        print(f"  Avg Latency:             {s['avg_total_ms']:.0f} ms total "
              f"({s['avg_retrieve_ms']:.0f} retrieve / {s['avg_generate_ms']:.0f} generate)")
        print(f"  Avg Best Retrieval Score:{s['avg_best_score']:.3f}")
        print(f"{'='*60}\n")

        if result["categories"]:
            print("Per-category breakdown:")
            print(f"  {'Category':<12} {'N':<3} {'Faith':<7} {'Rel':<6} {'Correct':<8} {'Comp':<6}")
            for cat, c in sorted(result["categories"].items()):
                print(f"  {cat:<12} {c['n']:<3} {c['avg_faithfulness']:<7.2f} "
                      f"{c['avg_relevancy']:<6.2f} {c['avg_answer_correctness']:<8.2f} {c['avg_completeness']:<6.2f}")
            print()

        if result["abstained"]:
            print("Abstained (low-confidence / no relevant info):")
            for q in result["abstained"]:
                print(f"  - {q}")
            print()

        if result["worst"]:
            print("Worst-performing queries:")
            for r in result["worst"]:
                print(f"  - [{r['category'] or 'general'}] {r['question'][:60]} "
                      f"(faith={r['faithfulness']}, rel={r['relevancy']})")
            print()

        if args.json:
            import json as j
            print(j.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Per-query breakdown:")
            print(f"  {'Question':<42} {'Cat':<10} {'Faith':<6} {'Rel':<6} {'Corr':<6} {'Comp':<6} {'ms':<8}")
            print(f"  {'-'*40}  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*6}")
            for r in result["results"]:
                q = r["question"][:40]
                f = "  -" if r["faithfulness"] is None else f"{r['faithfulness']:<6.1f}"
                rel = "  -" if r["relevancy"] is None else f"{r['relevancy']:<6.1f}"
                c = "  -" if r["answer_correctness"] is None else f"{r['answer_correctness']:<6.1f}"
                comp = "  -" if r["completeness"] is None else f"{r['completeness']:<6.2f}"
                print(f"  {q:<42} {(r['category'] or 'general'):<10} {f} {rel} {c} {comp} {r['total_ms']:<8.0f}")
    elif args.command == "up":
        import subprocess
        subprocess.run(["docker", "compose", "up", "-d"], check=True)
        print("Services started: Qdrant (6333), Neo4j (7474/7687)")
        print("Make sure Ollama is running separately (ollama serve)")
    elif args.command == "down":
        import subprocess
        subprocess.run(["docker", "compose", "down"], check=True)
        print("Services stopped.")
    elif args.command == "serve":
        import uvicorn
        uvicorn.run(
            "src.server:app",
            host=config.SERVER_HOST,
            port=config.SERVER_PORT,
            reload=False,
        )


if __name__ == "__main__":
    main()
