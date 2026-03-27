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