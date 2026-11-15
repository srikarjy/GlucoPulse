"""Shared adapter plumbing."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from serving.healthcare.deid import IdentifierRejected, validate_patient_id
from sources.model import Quarantined, RawMeasurement

CLIP = 64


@dataclass
class AdapterResult:
    raws: list[RawMeasurement] = field(default_factory=list)
    quarantined: list[Quarantined] = field(default_factory=list)
    stats: Counter = field(default_factory=Counter)


def checked_patient(patient: str) -> str:
    """Same rule as the FHIR facade: pseudonymous ids only, never a direct identifier."""
    try:
        return validate_patient_id(patient)
    except IdentifierRejected as exc:
        raise ValueError(str(exc)) from exc


def clip(value: Any) -> Any:
    """Untrusted text echoed into quarantine reports is bounded so reports can't balloon."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and abs(value) < 1e15:
        return value
    text = str(value) if not isinstance(value, str) else value
    return text if len(text) <= CLIP else text[:CLIP] + f"...(+{len(text) - CLIP} chars)"
