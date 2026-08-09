import json
import httpx
from src.helpers import config
from src.knowledge_graph.schemas import GraphEntity, GraphRelationship, GraphExtraction

BATCH_PROMPT = """Extract entities and relationships from each text chunk below.

Allowed entity types: Person, Organization, Location, Concept, Product
Allowed relationship types: WORKS_FOR, LOCATED_IN, RELATES_TO, PART_OF

Return a JSON array where each element corresponds to one text chunk:
[{{"nodes": [{{"name": str, "type": str, "description": str}}], "edges": [{{"source": str, "target": str, "relation": str, "description": str}}]}}]

Text chunks:
{chunks}

Return ONLY valid JSON, no additional text."""


class GraphExtractor:
    def __init__(self):
        self.client = httpx.Client(timeout=60.0)

    def __call__(self, texts: list[str]) -> list[GraphExtraction]:
        chunks_text = "\n---\n".join(f"Chunk {i}:\n{t[:1500]}" for i, t in enumerate(texts))
        prompt = BATCH_PROMPT.format(chunks=chunks_text)
        try:
            raw = self._call_llm(prompt)
            data = self._parse_response(raw)
            return [self._to_extraction(item) for item in data]
        except Exception as e:
            print(f"  KG batch failed: {e}")
            return [GraphExtraction() for _ in texts]

    def _call_llm(self, prompt: str) -> str:
        payload = {
            "model": config.LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 2048},
        }
        resp = self.client.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def _parse_response(self, raw: str):
        raw = raw.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        start = raw.find("[") if "[" in raw else raw.find("{")
        if start == -1:
            return {}
        end = raw.rfind("]") if "[" in raw else raw.rfind("}")
        raw = raw[start:end+1]
        return json.loads(raw)

    def _to_extraction(self, data: dict) -> GraphExtraction:
        entities = []
        for n in data.get("nodes", []):
            entities.append(GraphEntity(
                name=n.get("name", ""),
                type=n.get("type", "OTHER"),
                description=n.get("description", ""),
            ))
        relationships = []
        for e in data.get("edges", []):
            relationships.append(GraphRelationship(
                source=e.get("source", ""),
                target=e.get("target", ""),
                relation=e.get("relation", ""),
                description=e.get("description", ""),
            ))
        return GraphExtraction(entities=entities, relationships=relationships)


def extract_batch(texts: list[str]) -> list[GraphExtraction]:
    extractor = GraphExtractor()
    return extractor(texts)
