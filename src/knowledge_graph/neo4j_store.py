import re

from neo4j import GraphDatabase
from src.helpers import config
from src.knowledge_graph.schemas import GraphExtraction


_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )
    return _driver


def init_store(reset: bool = False):
    driver = _get_driver()
    with driver.session() as session:
        if reset:
            session.run("MATCH (n) DETACH DELETE n")
            print("Reset Neo4j graph")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)")
        session.run("CREATE FULLTEXT INDEX entity_names IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]")
        print("Neo4j store initialized")


def _batches(items: list, size: int = 1000):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def store_extractions(extractions: list[GraphExtraction]):
    entities = [e for ex in extractions for e in ex.entities]
    relationships = [r for ex in extractions for r in ex.relationships]

    if not entities and not relationships:
        return

    driver = _get_driver()
    with driver.session() as session:
        for batch in _batches(entities):
            session.run(
                "UNWIND $rows AS e "
                "MERGE (n:Entity {name: e.name}) "
                "SET n.type = e.type, n.description = e.description",
                rows=[
                    {"name": e.name, "type": e.type, "description": e.description}
                    for e in batch
                ],
            )
        for batch in _batches(relationships):
            session.run(
                "UNWIND $rows AS r "
                "MERGE (s:Entity {name: r.source}) "
                "MERGE (t:Entity {name: r.target}) "
                "MERGE (s)-[rel:RELATED {relation: r.relation}]->(t) "
                "SET rel.description = r.description",
                rows=[
                    {"source": r.source, "target": r.target, "relation": r.relation, "description": r.description}
                    for r in batch
                ],
            )


def query_related(entity_name: str, max_hops: int = 2, limit: int = 20) -> list[dict]:
    driver = _get_driver()
    max_hops = int(max(max_hops, 1))
    with driver.session() as session:
        result = session.run(
            f"MATCH (e:Entity {{name: $name}})-[r:RELATED*1..{max_hops}]-(related) "
            "RETURN DISTINCT related.name AS name, related.type AS type, "
            "related.description AS description, "
            "last(r).relation AS relation "
            "LIMIT $limit",
            name=entity_name,
            limit=limit,
        )
        return [dict(record) for record in result]


def query_entities_in_text(text: str) -> list[dict]:
    words = sorted({w for w in re.findall(r"[A-Za-z0-9-]+", text.lower()) if len(w) >= 2})[:100]
    if not words:
        return []

    driver = _get_driver()
    with driver.session() as session:
        result = session.run(
            "CALL db.index.fulltext.queryNodes('entity_names', $q) YIELD node "
            "RETURN node.name AS name, node.type AS type, node.description AS description",
            q=" OR ".join(words),
        )
        hits = [dict(record) for record in result]

    lowered = text.lower()
    return [h for h in hits if h["name"].lower() in lowered]


def close():
    global _driver
    if _driver:
        _driver.close()
        _driver = None
