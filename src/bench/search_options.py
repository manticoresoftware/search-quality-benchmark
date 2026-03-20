from __future__ import annotations


def normalize_search_options(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    if s.lower().startswith("option "):
        s = s[7:].strip()
    if s.endswith(";"):
        s = s[:-1].strip()
    return s

