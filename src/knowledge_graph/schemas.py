from dataclasses import dataclass, field


@dataclass
class GraphEntity:
    name: str
    type: str
    description: str = ""


@dataclass
class GraphRelationship:
    source: str
    target: str
    relation: str
    description: str = ""


@dataclass
class GraphExtraction:
    entities: list[GraphEntity] = field(default_factory=list)
    relationships: list[GraphRelationship] = field(default_factory=list)
