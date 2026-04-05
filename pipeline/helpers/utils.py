from .html_stripper import _HTMLStripper
import re

def strip_html(html: str) -> str:
    s = _HTMLStripper()
    s.feed(html)
    text = s.get_data()
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)

def normalize_dosing_text(text: str) -> str:
    text = re.sub(r'\s*[•·▪▸]\s*', '. ', text)
    text = re.sub(r'([A-Za-z\)]):(\s+[A-Z])', r'\1. \2', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()