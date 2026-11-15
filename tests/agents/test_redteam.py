"""Regression gate: every red-team attack on the triage panel must be blocked.
A failure here means a guardrail (injection quarantine, identifier/other-patient
filtering, tool scoping and budget, grounding, rule floor, future isolation) regressed."""
from __future__ import annotations

import os
import secrets

import pytest

os.environ.setdefault("CLINICAL_AGENT_JWT_SECRET", secrets.token_hex(32))

from agents.triage.redteam import ATTACKS, run_attack, run_all  # noqa: E402


@pytest.mark.parametrize("attack", ATTACKS, ids=[a.name for a in ATTACKS])
def test_attack_is_blocked(attack):
    ok, why = run_attack(attack)
    assert ok, f"{attack.category}: {why}"


def test_every_category_is_covered_and_all_blocked():
    rows = run_all()
    assert {"injection", "privacy", "scope", "grounding", "safety", "future"} <= {r[0] for r in rows}
    assert all(r[2] for r in rows), [r for r in rows if not r[2]]
