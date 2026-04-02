import re
from typing import Optional
from config import CFG
from .ingestion_constants import DOSE_PATTERNS, ROUTE_KEYWORDS, POPULATION_KEYWORDS

def parse_dose_value(text: str) -> tuple[Optional[float], Optional[str]]:
    for pattern in DOSE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1)), match.group(2).lower()
            except ValueError:
                pass
    return None, None

def parse_clinical_route(text: str) -> Optional[str]:
    lower_text = text.lower()
    for route, keywords in ROUTE_KEYWORDS.items():
        if any(k in lower_text for k in keywords):
            return route
    return None

def parse_patient_population(text: str) -> str:
    lower_text = text.lower()
    for pop, keywords in POPULATION_KEYWORDS.items():
        if any(k in lower_text for k in keywords):
            return pop
    return "general"

def execute_semantic_chunking(text: str, max_chars: int = None, overlap: int = None) -> list[str]:
    max_chars = max_chars or CFG.ingestion.max_chunk_chars
    overlap   = overlap   or CFG.ingestion.chunk_overlap

    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, buf = [], ""
    for sent in sentences:
        if len(buf) + len(sent) + 1 <= max_chars:
            buf = (buf + " " + sent).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = sent
    if buf:
        chunks.append(buf)

    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            prev_tail = chunks[i-1][-overlap:]
            chunk = prev_tail + " " + chunk
        overlapped_chunks.append(chunk.strip())
    return overlapped_chunks