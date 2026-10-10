"""
Pseudonymization and identifier screening.

Scope, stated plainly: AZT1D is published de-identified by its authors and
carries no names, MRNs, or contact data. This module does two narrow things:

1. `pseudonymize` -- keyed HMAC so the ids that leave the system (FHIR
   Patient.id, audit log) are not the source dataset's ids. Without the key
   the mapping can't be recomputed or reversed. Rotating the key
   re-pseudonymizes everything.
2. `reject_direct_identifiers` -- screens strings that arrive from callers
   (patient ids, free-text) for the identifier shapes HIPAA Safe Harbor
   (45 CFR 164.514(b)(2)) lists that are machine-detectable: email, phone,
   SSN, IP, URL, long numeric record numbers, dates.

It is NOT Safe Harbor de-identification of the dataset: forecasting needs
full timestamps, and Safe Harbor requires dates reduced to year. Data with
full dates is, at best, a HIPAA Limited Data Set (needs a data use agreement)
unless an Expert Determination says otherwise.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re

_PATIENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"),
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "url": re.compile(r"https?://\S+", re.I),
    "date": re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"),
    "long_number": re.compile(r"\b\d{7,}\b"),  # MRN / account / device serial shaped
}


class IdentifierRejected(ValueError):
    pass


def scan_text(text: str) -> list[str]:
    return [name for name, rx in _PATTERNS.items() if rx.search(text)]


def reject_direct_identifiers(value: str, field_name: str = "value") -> str:
    hits = scan_text(value)
    if hits:
        raise IdentifierRejected(f"{field_name} looks like it contains identifiers ({', '.join(hits)})")
    return value


def validate_patient_id(patient_id: str) -> str:
    if not _PATIENT_ID.match(patient_id):
        raise IdentifierRejected("patient id must be 1-64 chars of [A-Za-z0-9._-]")
    reject_direct_identifiers(patient_id, "patient id")
    return patient_id


def _key() -> bytes:
    key = os.getenv("GLUCOPULSE_PSEUDONYM_KEY", "")
    if len(key) < 32:
        raise RuntimeError("GLUCOPULSE_PSEUDONYM_KEY must be set (>= 32 chars) to pseudonymize ids")
    return key.encode()


def pseudonymize(raw_id: str) -> str:
    digest = hmac.new(_key(), raw_id.encode(), hashlib.sha256).hexdigest()
    return f"gp-{digest[:24]}"
