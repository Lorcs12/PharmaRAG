from .html_stripper import _HTMLStripper
import re

def strip_html(html: str) -> str:
    s = _HTMLStripper()
    s.feed(html)
    text = s.get_data()
    return re.sub(r'\s+', ' ', text).strip()