"""
Prompt-injection quarantine, ported from priorauth/criteria.py's
find_injections (same regex/shape) so both tool surfaces reject the same
patterns consistently. Free text (clinical notes, documents) is UNTRUSTED
DATA: anything matching this pattern is quarantined BEFORE an agent ever
sees it -- agents.mcp_server.tools.fhir_search_notes calls this itself,
the quarantine step is not something a caller could skip.
"""
from __future__ import annotations

import re

INJECTION_RE = re.compile(
    r"(ignore (all |any )?(previous|prior|policy|the)\b|disregard (all |any )?(previous|prior)|system override|"
    r"note to (the )?(ai|assistant|model)|you must (output|approve|mark)|###\s*instructions|"
    r"mark (this|the) (request|finding|note) as approved)",
    re.I)


def find_injections(texts: list[tuple[str, str]]) -> list[str]:
    """texts: [(ref, text)] -> refs of quarantined sources."""
    return [ref for ref, t in texts if INJECTION_RE.search(t or "")]
