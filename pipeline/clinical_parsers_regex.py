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

def execute_semantic_chunking(text: str, max_chars: int = None, overlap_sentences: int = 1) -> list[str]:
    max_chars = max_chars or CFG.ingestion.max_chunk_chars

    if len(text) <= max_chars:
        return [text]

    protected_text = re.sub(r'(\|[^\n]+\|\n)', r'\1<TBL_BREAK>', text)
    raw_sentences = re.split(r'(?<=[.!?])\s+', protected_text)
    sentences = [s.replace('<TBL_BREAK>', '').strip() for s in raw_sentences if s.strip()]
    merged_sentences: list[str] = []
    for sentence in sentences:
        if merged_sentences and merged_sentences[-1].strip().endswith(":"):
            merged_sentences[-1] = f"{merged_sentences[-1]} {sentence}".strip()
        else:
            merged_sentences.append(sentence)
    sentences = merged_sentences

    chunks: list[str] = []
    current_chunk_sents: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > max_chars and current_chunk_sents:
            chunks.append(" ".join(current_chunk_sents))
            overlap_start = max(0, len(current_chunk_sents) - overlap_sentences)
            current_chunk_sents = current_chunk_sents[overlap_start:]
            current_len = sum(len(s) + 1 for s in current_chunk_sents)
        current_chunk_sents.append(sent)
        current_len += sent_len + 1

    if current_chunk_sents:
        chunks.append(" ".join(current_chunk_sents))

    return chunks