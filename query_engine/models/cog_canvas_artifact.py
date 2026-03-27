
from dataclasses import dataclass, field
from .pharma_query_intent import PharmQueryIntent
from .retrieved_node import RetrievedNode


@dataclass
class CogCanvasArtifact:
    query:            str
    intent:           PharmQueryIntent
    verbatim_nodes:   list[RetrievedNode]
    paraphrase_nodes: list[RetrievedNode]
    causal_context:   list[RetrievedNode]
    macro_context:    list[RetrievedNode]
    table_references: list[RetrievedNode]
    total_latency_ms: float = 0.0
    conflicts:        list[dict] = field(default_factory=list)