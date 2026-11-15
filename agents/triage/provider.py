"""Read-only data for one triage case: the packet built by model/cgmacros/cases.py
(summaries of the 2 h before `as_of` + scenario forecasts). Outcome labels live in a
different file and are never loaded here, so agents cannot see the future."""
from __future__ import annotations

import json
from pathlib import Path

from agents.triage.guards import sanitize


class CaseProvider:
    def __init__(self, packet: dict):
        self.packet = packet
        self.patient = packet["patient"]
        self.quarantined: set[str] = set()     # paths of untrusted text replaced before any model saw it

    def _clean(self, obj, where: str):
        clean, flagged = sanitize(obj, where)
        self.quarantined.update(flagged)
        return clean

    @classmethod
    def from_file(cls, path: str | Path, case_id: str) -> "CaseProvider":
        return cls(json.loads(Path(path).read_text())[case_id])

    def check(self, patient: str) -> None:
        if patient != self.patient:
            raise PermissionError(f"this session is scoped to patient {self.patient}")

    def section(self, name: str) -> dict:
        return self._clean(self.packet[name], name)

    def scenarios(self, names: list[str]) -> list[dict]:
        menu = self.packet["scenarios"]
        unknown = [n for n in names if n not in menu]
        if unknown:
            raise KeyError(f"unknown scenarios {unknown}; available: {sorted(menu)}")
        return [self._clean(menu[n], f"scenarios.{n}") for n in names]

    def menu(self) -> dict[str, str]:
        return self._clean({n: s["description"] for n, s in self.packet["scenarios"].items()}, "scenarios.menu")
