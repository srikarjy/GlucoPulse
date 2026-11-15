"""Read-only data for one triage case: the packet built by model/cgmacros/cases.py
(summaries of the 2 h before `as_of` + scenario forecasts). Outcome labels live in a
different file and are never loaded here, so agents cannot see the future."""
from __future__ import annotations

import json
from pathlib import Path


class CaseProvider:
    def __init__(self, packet: dict):
        self.packet = packet
        self.patient = packet["patient"]

    @classmethod
    def from_file(cls, path: str | Path, case_id: str) -> "CaseProvider":
        return cls(json.loads(Path(path).read_text())[case_id])

    def check(self, patient: str) -> None:
        if patient != self.patient:
            raise PermissionError(f"this session is scoped to patient {self.patient}")

    def section(self, name: str) -> dict:
        return self.packet[name]

    def scenarios(self, names: list[str]) -> list[dict]:
        menu = self.packet["scenarios"]
        unknown = [n for n in names if n not in menu]
        if unknown:
            raise KeyError(f"unknown scenarios {unknown}; available: {sorted(menu)}")
        return [menu[n] for n in names]

    def menu(self) -> dict[str, str]:
        return {n: s["description"] for n, s in self.packet["scenarios"].items()}
