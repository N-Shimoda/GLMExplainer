import re


def _normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\.$", "", s)
    return s.lower()
