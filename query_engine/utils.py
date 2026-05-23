import re

from .constants import (
    DOSE_CLAIM_RE, XML_WRAPPER_RE, INSUFF_RE,
    INSUFFICIENT_EVIDENCE_EXACT, INSUFF_SENTENCE_RE,
)


def generate_ngrams(text: str, n: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]

def extract_surrounding_context(text: str, urn: str, window: int = 300) -> str:
    idx = text.find(urn)
    if idx == -1:
        return ""
    start = max(0, idx - window)
    end   = min(len(text), idx + window)
    return text[start:end]


def normalize_llm_output(text: str) -> str:
    m = XML_WRAPPER_RE.match(text)
    if m:
        text = m.group(1).strip()

    if INSUFF_RE.search(text):
        cleaned = INSUFF_SENTENCE_RE.sub('', text).strip()
        if len(cleaned.strip('.,;: \n')) <= 40:
            return INSUFFICIENT_EVIDENCE_EXACT
        text = cleaned

    sentences = re.split(r'(?<=[.!?])\s+', text)
    clean: list[str] = []
    for idx, sent in enumerate(sentences):
        matches = list(DOSE_CLAIM_RE.finditer(sent))
        if not matches:
            clean.append(sent)
            continue

        prev_sent = sentences[idx - 1] if idx > 0 else ""
        next_sent = sentences[idx + 1] if idx + 1 < len(sentences) else ""
        neighborhood = f"{prev_sent} {sent} {next_sent}"

        orphaned = False
        for m in matches:
            start = max(0, m.start() - 300)
            end   = min(len(sent), m.end() + 300)
            has_local_source = "[SOURCE:" in sent[start:end]
            has_adjacent_source = "[SOURCE:" in neighborhood
            if not has_local_source and not has_adjacent_source:
                orphaned = True
                break
        if not orphaned:
            clean.append(sent)

    return " ".join(clean).strip()
