import json
import re
import time
import httpx
from src.controllers.hybrid_query import query as query_controller
from src.helpers import config

CONTEXT_JUDGE_PROMPT = """You are a strict judge for a retrieval-augmented QA system. Answer with ONLY a JSON object and nothing else.

Question: {question}

Retrieved context:
{context}

Generated answer:
{answer}

Score each metric as an integer 0-5:
- faithfulness: how fully the generated answer is supported by the retrieved context (no unsupported or contradictory claims). 5 = fully supported, 0 = completely unsupported.
- relevancy: how directly the generated answer addresses the question. 5 = complete and precise, 0 = irrelevant.
- context_relevance: how relevant and sufficient the retrieved context is for answering the question (independent of the answer). 5 = context directly answers the question, 0 = context is irrelevant.

Also set retrieval_support to 1 if the retrieved context contains the information needed to answer the question, else 0.

Return JSON like: {{"faithfulness": 4, "relevancy": 3, "context_relevance": 4, "retrieval_support": 1}}"""

GT_JUDGE_PROMPT = """You are a strict judge comparing an answer to a reference answer. Answer with ONLY a JSON object and nothing else.

Question: {question}

Reference answer:
{reference}

Generated answer:
{answer}

Score answer_correctness as an integer 0-5 for how well the generated answer matches the reference answer semantically (correctness, not wording).

For completeness, list up to 5 key facts (short strings) contained in the reference answer, and mark whether the generated answer covers each one.

Return JSON like: {{"answer_correctness": 4, "key_facts": ["fact1", "fact2"], "covered": [true, true]}}"""


def _call_llm(prompt: str, num_predict: int = 256) -> str:
    payload = {
        "model": config.JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": num_predict},
    }
    resp = httpx.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _extract_json_object(raw: str):
    raw = raw.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None


def _parse_score(raw: str) -> float:
    for ch in raw:
        if ch.isdigit():
            return float(ch)
    return 0.0


def _score(data: dict, key: str, default: float = 0.0) -> float:
    value = data.get(key)
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _judge_context(question: str, context: str, answer: str) -> dict:
    prompt = CONTEXT_JUDGE_PROMPT.format(question=question, context=context[:12000], answer=answer[:3000])
    values = {"faithfulness": [], "relevancy": [], "context_relevance": [], "retrieval_support": []}
    for _ in range(max(1, config.EVAL_JUDGE_PASSES)):
        try:
            data = _extract_json_object(_call_llm(prompt))
        except Exception:
            data = None
        if data:
            for key in values:
                values[key].append(_score(data, key))
    out = {}
    for key, vals in values.items():
        out[key] = sum(vals) / len(vals) if vals else 0.0
    return out


def _judge_ground_truth(question: str, reference: str, answer: str) -> dict:
    prompt = GT_JUDGE_PROMPT.format(question=question, reference=reference[:3000], answer=answer[:3000])
    correctness = []
    completeness = []
    for _ in range(max(1, config.EVAL_JUDGE_PASSES)):
        try:
            data = _extract_json_object(_call_llm(prompt))
        except Exception:
            data = None
        if data:
            correctness.append(_score(data, "answer_correctness"))
            facts = data.get("key_facts")
            covered = data.get("covered")
            if isinstance(facts, list) and isinstance(covered, list) and facts:
                comp = sum(
                    1.0 for f, c in zip(facts, covered) if c
                ) / len(facts)
                completeness.append(comp)
    return {
        "answer_correctness": sum(correctness) / len(correctness) if correctness else 0.0,
        "completeness": sum(completeness) / len(completeness) if completeness else 0.0,
    }


def _timing(trace: list | None, total_ms: float) -> dict:
    generate_ms = None
    if trace:
        for step in trace:
            if step.get("step") == "generate" and "elapsed_ms" in step:
                generate_ms = step["elapsed_ms"]
    return {
        "total_ms": total_ms,
        "retrieve_ms": (total_ms - generate_ms) if generate_ms is not None else None,
        "generate_ms": generate_ms,
    }


def evaluate_query(question: str, reference: str | None = None,
                   category: str = "") -> dict:
    trace = []
    start = time.time()
    result = query_controller(question, top_k=5, trace=trace)
    total_ms = round((time.time() - start) * 1000, 1)

    confident = result.get("confident", True)
    results = result.get("results", [])
    answer = result.get("answer", "")
    context = "\n\n".join(
        f"[{r['metadata'].get('filename', 'unknown')}] {r['text'][:500]}"
        for r in results
    )
    sources = sorted(set(r["metadata"].get("filename", "unknown") for r in results))

    abstained = (not confident) or not results
    timings = _timing(trace, total_ms)

    entry = {
        "question": question,
        "category": category,
        "answer": answer,
        "sources": sources,
        "confident": confident,
        "abstained": abstained,
        "faithfulness": None,
        "relevancy": None,
        "context_relevance": None,
        "retrieval_support": None,
        "answer_correctness": None,
        "completeness": None,
        "abstained_correctly": None,
        "best_score": results[0]["score"] if results else None,
        **timings,
    }

    if category == "negative":
        entry["abstained_correctly"] = abstained

    if abstained or not context.strip():
        return entry

    ctx = _judge_context(question, context, answer)
    entry["faithfulness"] = ctx["faithfulness"]
    entry["relevancy"] = ctx["relevancy"]
    entry["context_relevance"] = ctx["context_relevance"]
    entry["retrieval_support"] = ctx["retrieval_support"]

    if reference:
        gt = _judge_ground_truth(question, reference, answer)
        entry["answer_correctness"] = gt["answer_correctness"]
        entry["completeness"] = gt["completeness"]

    return entry


def _avg(values: list[float]) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def evaluate(test_file: str) -> dict:
    with open(test_file) as f:
        raw_queries = json.load(f)

    queries = []
    for q in raw_queries:
        if isinstance(q, str):
            queries.append({"question": q})
        else:
            queries.append(q)

    results = []
    for i, q in enumerate(queries, 1):
        question = q["question"]
        reference = q.get("answer")
        category = q.get("category", "")
        print(f"  [{i}/{len(queries)}] [{category or 'general':<9}] {question[:55]}...")
        start = time.time()
        try:
            r = evaluate_query(question, reference=reference, category=category)
            elapsed = time.time() - start
            print(f"    {elapsed:5.1f}s  confident={r['confident']}  "
                  f"faith={r['faithfulness']}  rel={r['relevancy']}  "
                  f"correct={r['answer_correctness']}  comp={r['completeness']}")
            results.append(r)
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({
                "question": question, "category": category, "answer": "",
                "sources": [], "confident": False, "abstained": True,
                "faithfulness": None, "relevancy": None, "context_relevance": None,
                "retrieval_support": None, "answer_correctness": None,
                "completeness": None, "abstained_correctly": None,
                "best_score": None, "total_ms": 0.0, "retrieve_ms": None,
                "generate_ms": None, "error": str(e),
            })

    n = len(results)
    num_gt = sum(1 for r in results if r["answer_correctness"] is not None)
    abstained = [r for r in results if r["abstained"]]
    negative = [r for r in results if r["category"] == "negative"]

    summary = {
        "num_queries": n,
        "num_with_ground_truth": num_gt,
        "avg_faithfulness": _avg([r["faithfulness"] for r in results]),
        "avg_relevancy": _avg([r["relevancy"] for r in results]),
        "avg_context_relevance": _avg([r["context_relevance"] for r in results]),
        "retrieval_support_rate": _avg([r["retrieval_support"] for r in results]),
        "avg_answer_correctness": _avg([r["answer_correctness"] for r in results]),
        "avg_completeness": _avg([r["completeness"] for r in results]),
        "abstention_rate": len(abstained) / n if n else 0.0,
        "abstained_correctly_rate": _avg([r["abstained_correctly"] for r in negative]) if negative else None,
        "avg_total_ms": _avg([r["total_ms"] for r in results]),
        "avg_retrieve_ms": _avg([r["retrieve_ms"] for r in results]),
        "avg_generate_ms": _avg([r["generate_ms"] for r in results]),
        "avg_best_score": _avg([r["best_score"] for r in results]),
        "max_best_score": max((r["best_score"] for r in results if r["best_score"] is not None), default=0.0),
    }

    categories = {}
    for r in results:
        cat = r["category"] or "general"
        cats = categories.setdefault(cat, {"n": 0, "faithfulness": [], "relevancy": [],
                                           "answer_correctness": [], "completeness": []})
        cats["n"] += 1
        cats["faithfulness"].append(r["faithfulness"])
        cats["relevancy"].append(r["relevancy"])
        cats["answer_correctness"].append(r["answer_correctness"])
        cats["completeness"].append(r["completeness"])
    categories = {
        cat: {
            "n": c["n"],
            "avg_faithfulness": _avg(c["faithfulness"]),
            "avg_relevancy": _avg(c["relevancy"]),
            "avg_answer_correctness": _avg(c["answer_correctness"]),
            "avg_completeness": _avg(c["completeness"]),
        }
        for cat, c in categories.items()
    }

    scored = [r for r in results if r["faithfulness"] is not None and r["relevancy"] is not None]
    scored.sort(key=lambda r: -(r["faithfulness"] + r["relevancy"]))

    return {
        "summary": summary,
        "categories": categories,
        "worst": scored[-3:][::-1] if scored else [],
        "abstained": [r["question"] for r in abstained],
        "results": results,
    }
